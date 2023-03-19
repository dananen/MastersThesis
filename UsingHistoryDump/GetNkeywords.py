import osmium
import time
import pandas as pd
from UsingMainAPI import MainAPIHelperFunctions as mainhf
import HistoryHelperFunctions as hhf

KEYWORDS = mainhf.get_top_12_keywords()

class UserHistoryHandler(osmium.SimpleHandler):

    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.user_data = hhf.init_user_dict()
        self.prev_tags = {}
        self.counter = 0


    def process_element(self, element):
        self.counter += 1
        if element.uid in self.user_data.keys():
           for user_cs_dict in self.user_data[element.uid].values():
               if element.timestamp < user_cs_dict['cs_created_at']:
                    user_cs_dict['nkeywords'] += hhf.nbr_keywords_added(self.prev_tags, element.tags, KEYWORDS)
                  #  if nkeywords_added > 0:
                  #     print('Nbr keywords before update: {0}. ID: {1}, uid: {2}, version: {3}'.format(user_cs_dict['nkeywords'], element.id, element.uid, element.version))
                  #  user_cs_dict['nkeywords'] += nkeywords_added
                  #  if nkeywords_added > 0:
                  #     print('Nbr keywords after update: {0}.'.format(user_cs_dict['nkeywords']))

        self.prev_tags = {tag.k: tag.v for tag in element.tags}
        if self.counter % 1000000 == 0:
            print('Count: {0}, Time: {1}'.format(self.counter, round(time.perf_counter()-t1)))

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
    input_file = '/Users/dansvenonius/Desktop/Misc output/Reverted Data/elements_for_nkeywords.osm.pbf'
    output_csv = '/Users/dansvenonius/Desktop/History output/Reverted Data/user_data_w_nkeywords.csv'
else:
    input_file = '/Users/dansvenonius/Desktop/Misc output/Not Reverted Data/elements_for_nkeywords.osm.pbf'
    output_csv = '/Users/dansvenonius/Desktop/History output/Not Reverted Data/user_data_w_nkeywords.csv'

t1 = time.perf_counter()
uhh = UserHistoryHandler()
#uhh.apply_file(input_file)

big_dict = {}
for user_dict in uhh.user_data.values():
    for cs_dict in user_dict.values():
        big_dict = mainhf.add_dict(big_dict, cs_dict)
pd.DataFrame.from_dict(big_dict).to_csv(output_csv, index=False)
