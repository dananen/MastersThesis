import osmium
import time
import pandas as pd
from UsingMainAPI import MainAPIHelperFunctions as mainhf
import HistoryHelperFunctions as hhf
import gc

KEYWORDS = mainhf.get_top_12_keywords()

class UserHistoryHandler(osmium.SimpleHandler):

    def __init__(self, user_data_file):
        osmium.SimpleHandler.__init__(self)
        self.user_data = hhf.init_user_dict(user_data_file)
        self.prev_tag_keys = {}
        self.counter = 0
        self.ten_mil_counter = 0
        self.prev_el_version = 0
        self.prev_el_id = 0

    def process_element(self, element):
        self.counter += 1
        curr_tags = {tag.k: tag.v for tag in element.tags if tag.k in KEYWORDS}
        if len(curr_tags) > 0 and element.uid in self.user_data.keys():
            for user_cs_dict in self.user_data[element.uid].values():
                if element.timestamp >= user_cs_dict['cs_created_at']:
                    continue
                if not (element.version == self.prev_el_version+1 and self.prev_el_id == element.id):
                    self.prev_tag_keys.clear()
                for keyword, value in hhf.nbr_keywords_added(self.prev_tag_keys, curr_tags).items():
                    user_cs_dict['create_' + keyword] += value

        self.prev_tag_keys, self.prev_el_version, self.prev_el_id = curr_tags, element.version, element.id
        if self.counter % 10**6 == 0:
            gc.collect()
            self.counter = 0
            self.ten_mil_counter += 1
            print('ID: {0}, Count: {1}, Time: {2}'.format(element.id, self.ten_mil_counter*10**6, round(time.perf_counter()-t1)))

    def node(self, node):
        self.process_element(node)

    def way(self, way):
        self.process_element(way)

    def relation(self, relation):
        self.process_element(relation)


########### INPUT #############
reverted_data = True
###############################

if reverted_data:
    # 700 miljoner element, cirkus (277n, 428w, 20r)
    input_file = '/Users/dansvenonius/Desktop/Misc output/Reverted Data/elements_touched_by_users.osm.pbf'
    input_user_data = '/Users/dansvenonius/Desktop/History output/Reverted Data/user_data_w_operations.csv'
    output_csv = '/Users/dansvenonius/Desktop/History output/Reverted Data/user_data_final.csv'
else:
    # 1100 miljoner element, ungefär (215n, 872w, 26r)
    input_file = '/Users/dansvenonius/Desktop/Misc output/Not Reverted Data/elements_touched_by_users.osm.pbf'
    input_user_data = '/Users/dansvenonius/Desktop/History output/Not Reverted Data/user_data_w_operations.csv'
    output_csv = '/Users/dansvenonius/Desktop/History output/Not Reverted Data/user_data_final.csv'

#input_file = '/Users/dansvenonius/Desktop/test_element_tags.xml'

t1 = time.perf_counter()
uhh = UserHistoryHandler(input_user_data)
uhh.apply_file(input_file)

big_dict = {}
for user_dict in uhh.user_data.values():
    for cs_dict in user_dict.values():
        big_dict = mainhf.add_dict(big_dict, cs_dict)
pd.DataFrame.from_dict(big_dict).to_csv(output_csv, index=False)
