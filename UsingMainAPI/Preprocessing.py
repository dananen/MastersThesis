import osmapi
import pandas as pd
import time
import MainAPIHelperFunctions

api = osmapi.OsmApi()

########### INPUT ############
reverted_data = False
##############################

if reverted_data:
    input_file = '/Users/dansvenonius/Desktop/Misc output/Reverted Data/reverted_changesets.csv'
    output_path = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/'
else:
    input_file = '/Users/dansvenonius/Desktop/Misc output/Not Reverted Data/false_label_changesets.csv'
    output_path = '/Users/dansvenonius/Desktop/Preprocessing output/Not Reverted Data/'

cs_df = pd.read_csv(input_file)
VALID_TAGS = MainAPIHelperFunctions.get_valid_tags()

bad_ids = set()
big_changeset_data = {}  # ready data
big_node_create_dict, big_way_create_dict, big_relation_create_dict, big_user_dict = {}, {}, {}, {}  # to be processed further
big_node_dict, big_way_dict, big_relation_dict = {}, {}, {}  # to be processed further

tstart = time.perf_counter()
for idx, row in cs_df.iterrows():
    if reverted_data:
        id, cs_created_at = row['reverted_id'], row['reverted_created_at']
    else:
        id, cs_created_at = row['cs_id'], row['created_at']
    if id in bad_ids:
        continue
    try:
        uid = api.ChangesetGet(id)['uid']
    except:
        print("\nSkipped CS {0}. Couldn't retrieve user id.".format(id))
        bad_ids.add(id)
        continue

    t1 = time.perf_counter()
    cs = api.ChangesetGet(id)
    osmchange = api.ChangesetDownload(id)
    if len(osmchange) == 0:
        bad_ids.add(id)
        continue

    changeset_data, dict_list = MainAPIHelperFunctions.osmchange_to_changeset_data(id, cs_created_at, cs, osmchange, VALID_TAGS)
    if not changeset_data and not dict_list:
        bad_ids.add(id)
        continue
    big_changeset_data = MainAPIHelperFunctions.add_dict(big_changeset_data, changeset_data)
    big_user_dict = MainAPIHelperFunctions.add_dict(big_user_dict,
                                             {'uid': uid,
                                              'cs_id': id,
                                              'cs_created_at': cs_created_at})
    for element_dict in dict_list:
        if element_dict['version'] == 1:
            if element_dict['type'] == 'node':
                big_node_create_dict = MainAPIHelperFunctions.add_dict(big_node_create_dict, element_dict)
            elif element_dict['type'] == 'way':
                big_way_create_dict = MainAPIHelperFunctions.add_dict(big_way_create_dict, element_dict)
            elif element_dict['type'] == 'relation':
                big_relation_create_dict = MainAPIHelperFunctions.add_dict(big_relation_create_dict, element_dict)
        else:
            if element_dict['type'] == 'node':
                big_node_dict = MainAPIHelperFunctions.add_dict(big_node_dict, element_dict)
            elif element_dict['type'] == 'way':
                big_way_dict = MainAPIHelperFunctions.add_dict(big_way_dict, element_dict)
            elif element_dict['type'] == 'relation':
                big_relation_dict = MainAPIHelperFunctions.add_dict(big_relation_dict, element_dict)
    t2 = time.perf_counter()
    print('Time for changeset {0} of {1}: {2} seconds'.format(idx+1, len(cs_df.axes[0]), round(t2-t1, 3)))
    time.sleep(1)

changeset_df = pd.DataFrame.from_dict(big_changeset_data).drop_duplicates()
user_df = pd.DataFrame.from_dict(big_user_dict).drop_duplicates()
node_df = pd.DataFrame.from_dict(big_node_dict).drop_duplicates()
way_df = pd.DataFrame.from_dict(big_way_dict).drop_duplicates()
relation_df = pd.DataFrame.from_dict(big_relation_dict).drop_duplicates()
node_create_df = pd.DataFrame.from_dict(big_node_create_dict).drop_duplicates()
way_create_df = pd.DataFrame.from_dict(big_way_create_dict).drop_duplicates()
relation_create_df = pd.DataFrame.from_dict(big_relation_create_dict).drop_duplicates()
bad_ids_df = pd.DataFrame.from_dict({'uid': list(bad_ids)}).drop_duplicates()

changeset_df.to_csv(output_path + 'changeset_data.csv', index=False)
user_df.to_csv(output_path + 'user_data.csv', index=False)
node_df.to_csv(output_path + 'node_data.csv', index=False)
way_df.to_csv(output_path + 'way_data.csv', index=False)
relation_df.to_csv(output_path + 'relation_data.csv', index=False)
node_create_df.to_csv(output_path + 'node_create_data.csv', index=False)
way_create_df.to_csv(output_path + 'way_create_data.csv', index=False)
relation_create_df.to_csv(output_path + 'relation_create_data.csv', index=False)
bad_ids_df.to_csv(output_path + 'bad_uids.csv', index=False)

tend = time.perf_counter()
print('\nTime for code to run: {0} seconds'.format(round(tend-tstart, 2)))