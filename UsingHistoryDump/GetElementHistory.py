import osmium
import time
import HistoryHelperFunctions as hhf
import pandas as pd
from UsingMainAPI import MainAPIHelperFunctions as mainhf

VALID_TAGS = mainhf.get_valid_tags()


class HistoryHandler(osmium.SimpleHandler):
    def __init__(self, input_path):
        osmium.SimpleHandler.__init__(self)
        self.pre_node_data = hhf.init_big_element_dict(input_path, 'node')
        self.pre_way_data = hhf.init_big_element_dict(input_path, 'way')
        self.pre_relation_data = hhf.init_big_element_dict(input_path, 'relation')
        self.proc_elements = {}
        self.bad_elements = {'cs_id': [], 'type': [], 'element_id': []}

        self.element_history = []
        self.reverted_versions = []
        self.prev_element_id = -1
        self.prev_type = ''

    def get_pre_data(self, type):
        if type == 'node':
            return self.pre_node_data
        elif type == 'way':
            return self.pre_way_data
        elif type == 'relation':
            return self.pre_relation_data

    def add_element(self, element, type):
        if self.element_history and self.prev_element_id != element.id:
            processed_history = hhf.process_element_history(self.element_history, self.get_pre_data(self.prev_type)[self.prev_element_id])
            for element_dict in processed_history.values():
                if len(element_dict) == 1:
                    self.bad_elements['cs_id'].append(element_dict['cs_id'])
                    self.bad_elements['element_id'].append(self.prev_element_id)
                    self.bad_elements['type'].append(self.prev_type)
                else:
                    self.proc_elements = mainhf.add_dict(self.proc_elements, element_dict)
            del self.get_pre_data(self.prev_type)[self.prev_element_id]
            self.element_history.clear()
            self.prev_element_id = element.id
            self.prev_type = type
            self.reverted_versions = []

        if not self.reverted_versions:
            for nested_element_dict in self.get_pre_data(type)[element.id].values():
                for reverted_version in nested_element_dict['version']:
                    self.reverted_versions.append(reverted_version)

        if (self.prev_element_id == -1 or self.prev_element_id == element.id) and element.version <= max(self.reverted_versions):
            self.element_history.append(hhf.process_element_iteration(element, self.reverted_versions, VALID_TAGS))
        self.prev_element_id = element.id
        self.prev_type = type

    def node(self, node):
        self.add_element(node, 'node')

    def way(self, way):
        self.add_element(way, 'way')

    def relation(self, relation):
        self.add_element(relation, 'relation')


########### INPUT #############
reverted_data = True
###############################

if reverted_data:
    input_file = '/Users/dansvenonius/Desktop/Misc output/Reverted Data/reverted_elements_history.osm.pbf'
    input_path = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/'
    output_path = '/Users/dansvenonius/Desktop/History output/Reverted Data/'
else:
    input_file = '/Users/dansvenonius/Desktop/Misc output/Not Reverted Data/not_reverted_elements_history.osm.pbf'
    input_path = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/'
    output_path = '/Users/dansvenonius/Desktop/History output/Not Reverted Data/'

t1 = time.perf_counter()
hh = HistoryHandler(input_path)
print('Time for init: {0} seconds'.format(round(time.perf_counter() - t1, 2)))

hh.apply_file(input_file)

processed_elements_df = pd.DataFrame.from_dict(hh.proc_elements)
processed_elements_df.to_csv(output_path + 'elements_modifydelete_data.csv', index=False)
bad_elements_df = pd.DataFrame.from_dict(hh.bad_elements)
bad_elements_df.to_csv(output_path + 'bad_elements_data.csv', index=False)

print('Time to run code: {0} seconds'.format(round(time.perf_counter()-t1, 2)))


