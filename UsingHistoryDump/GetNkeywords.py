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

    def process_element(self, element):
        self.counter += 1
        if len(element.tags) > 0:
            curr_tags = {tag.k: tag.v for tag in element.tags if tag.k in KEYWORDS}
        else:
            curr_tags = {}
        if element.uid in self.user_data.keys():
            for user_cs_dict in self.user_data[element.uid].values():
               if element.timestamp < user_cs_dict['cs_created_at']:
                    if element.version == 1:
                        self.prev_tag_keys.clear()
                    for keyword, value in hhf.nbr_keywords_added(self.prev_tag_keys, curr_tags).items():
                        user_cs_dict['create_' + keyword] += value

        self.prev_tag_keys = curr_tags
        if self.counter % 10**6 == 0:
            gc.collect()
            self.counter = 0
            self.ten_mil_counter += 1
            print('Count: {0}, Time: {1}'.format(self.ten_mil_counter*10**6, round(time.perf_counter()-t1)))

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
    # 2.6 miljarder noder, 0.4 miljarder ways, 0.025 miljarder relations. ish 3 miljarder element.
    # 110 sekunder per 1 miljon element ger 110*3000 = 4 dygn ish...
    input_file = '/Users/dansvenonius/Desktop/Misc output/Reverted Data/elements_touched_by_users.osm.pbf'
    input_user_data = '/Users/dansvenonius/Desktop/History output/Reverted Data/user_data_w_operations.csv'
    output_csv = '/Users/dansvenonius/Desktop/History output/Reverted Data/user_data_final.csv'
else:
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
