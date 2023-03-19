import pandas as pd

def element_ids_to_txt(input_path, output_file):
    ids_list = []
    node_ids = pd.read_csv(input_path + 'node_data.csv').loc[:, 'element_id']
    for _, id in node_ids.items():
        ids_list.append(('n', str(id)))

    way_ids = pd.read_csv(input_path + 'way_data.csv').loc[:, 'element_id']
    for _, id in way_ids.items():
        ids_list.append(('w', str(id)))

    relation_ids = pd.read_csv(input_path + 'relation_data.csv').loc[:, 'element_id']
    for _, id in relation_ids.items():
        ids_list.append(('r', str(id)))

    with open(output_file, 'w') as f:
        for letter, id in ids_list:
            f.write(letter + id + '\n')


########## INPUT ###########
reverted_data = True
############################

if reverted_data:
    input_path = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/'
    output_file = '/Users/dansvenonius/Desktop/Misc output/Reverted Data/CLI_element_ids_for_history.txt'
else:
    input_path = '/Users/dansvenonius/Desktop/Preprocessing output/Not Reverted Data/'
    output_file = '/Users/dansvenonius/Desktop/Misc output/Not Reverted Data/CLI_element_ids_for_history.txt'

element_ids_to_txt(input_path, output_file)