import osmium
import time
import pandas as pd
from UsingMainAPI import MainAPIHelperFunctions as mainhf
import HistoryHelperFunctions as hhf
import gc

class UserHistoryHandler(osmium.SimpleHandler):

    def __init__(self, input_user_data_file, file_writer):
        osmium.SimpleHandler.__init__(self)
        self.user_data = hhf.init_user_dict(input_user_data_file)
        self.relevant_ids = set()
        self.ten_mil_count, self.sub_count = 0, 0
        self.prev_element_visible = False
        self.file_writer = file_writer

    def process_element(self, element, type):
        self.sub_count += 1
        if element.uid in self.user_data.keys():
            for user_cs_dict in self.user_data[element.uid].values():
                if element.timestamp < user_cs_dict['cs_created_at']:
                    user_cs_dict[hhf.determine_operation(self.prev_element_visible, element.visible, version=element.version)] += 1
                    self.relevant_ids.add(type + str(element.id))
        self.prev_element_visible = element.visible
        if self.sub_count % 10**7 == 0:
            self.ten_mil_count += 1
            self.sub_count = 0
            for id in self.relevant_ids:
                self.file_writer.write(id + '\n')
            print('Type: {0} ID: {1}, Count: {2} Time: {3}, relevant_ids length: {4}'.format(type, element.id,
                                    self.ten_mil_count*10**7, round(time.perf_counter()-t1), len(self.relevant_ids)))
            self.relevant_ids.clear()
            gc.collect()

    def node(self, node):
        self.process_element(node, 'n')

    def way(self, way):
        self.process_element(way, 'w')

    def relation(self, relation):
        self.process_element(relation, 'r')



########### INPUT #############
reverted_data = True
###############################

input_file = '/Users/dansvenonius/Desktop/history-230227.osm.pbf'
#input_file = '/Users/dansvenonius/Desktop/test_element_history.xml'
if reverted_data:
    input_user_data_file = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/user_data.csv'
    output_csv = '/Users/dansvenonius/Desktop/History output/Reverted Data/user_data_w_operations.csv'
    output_txt = '/Users/dansvenonius/Desktop/History output/Reverted Data/CLI_element_ids_for_nkeywords.txt'
else:
    input_user_data_file = '/Users/dansvenonius/Desktop/Preprocessing output/Not Reverted Data/user_data.csv'
    output_csv = '/Users/dansvenonius/Desktop/History output/Not Reverted Data/user_data_w_operations.csv'
    output_txt = '/Users/dansvenonius/Desktop/History output/Not Reverted Data/CLI_element_ids_for_nkeywords.txt'

t1 = time.perf_counter()

with open(output_txt, 'w') as f:
    uhh = UserHistoryHandler(input_user_data_file, f)
    uhh.apply_file(input_file)

big_dict = {}
for user_dict in uhh.user_data.values():
    for cs_dict in user_dict.values():
        cs_dict['contributions'] = cs_dict['create'] + cs_dict['modify'] + cs_dict['delete']
        big_dict = mainhf.add_dict(big_dict, cs_dict)
pd.DataFrame.from_dict(big_dict).to_csv(output_csv, index=False)



