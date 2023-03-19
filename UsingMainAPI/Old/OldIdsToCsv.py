import osmapi
from OldCsIdToData import cs_id_to_data
import MainAPIHelperFunctions
import time
import pandas as pd
import numpy as np

VALID_TAGS = MainAPIHelperFunctions.get_valid_tags()
EDITOR_APPS = MainAPIHelperFunctions.get_editor_apps()
api = osmapi.OsmApi()

ids_df = pd.read_csv('/Users/dansvenonius/Desktop/test_revert_data.csv')
#ids_df = pd.read_csv('/Users/dansvenonius/Desktop/revert_data.csv')
seed = 1
#seed = None
nbr_samples = min(3, len(ids_df.axes[0]))

np.random.seed(seed)

indices = np.random.choice(len(ids_df.axes[0]), nbr_samples, replace=False)
reverted_ids = ids_df.loc[indices]['reverted_id']
revert_datetimes = ids_df.loc[indices]['created_at'].apply(MainAPIHelperFunctions.string_to_datetime)

tstart = time.perf_counter()

changeset_big_dict = {}
user_big_dict = {}
feature_big_dict = {}

user_data_acquired = {}
bad_ids = set()
query_user = True

#reverted_ids = pandas.Series([42277034])

for i, (idx, id) in enumerate(reverted_ids.items()):
    try:
        uid = api.ChangesetGet(id)['uid']
    except:
        print("Skipped CS {0}. Couldn't retrieve user id.".format(id))
        continue

    if uid not in bad_ids:
        query_user = uid not in user_data_acquired.keys()
        #query_user = False
        t1 = time.perf_counter()
        print("\nParsing id: " + str(id))
        changeset_data, user_data, feature_data_list = cs_id_to_data(id, api, VALID_TAGS, EDITOR_APPS,
                                                                     query_user_data=query_user,
                                                                     max_cs_history=100,
                                                                     max_cs_objects=1000,
                                                                     max_nchanges=50000)

        if not changeset_data and not user_data and not feature_data_list:
            bad_ids.add(uid)
            continue
        if query_user:
            user_data_acquired[user_data['id']] = user_data

        changeset_big_dict = MainAPIHelperFunctions.add_dict(changeset_big_dict, changeset_data)
        user_big_dict = MainAPIHelperFunctions.add_dict(user_big_dict, user_data_acquired[uid])
        for feature_dict in feature_data_list:
            feature_big_dict = MainAPIHelperFunctions.add_dict(feature_big_dict, feature_dict)

        t2 = time.perf_counter()
        print("Time for datapoint {0} of {1}: {2} seconds".format(i+1, reverted_ids.size, round(t2-t1, 2)))

tend = time.perf_counter()
print("Total time to run code: {0}".format(tend-tstart))


changeset_df = pd.DataFrame.from_dict(changeset_big_dict)
user_df = pd.DataFrame.from_dict(user_big_dict)
feature_df = pd.DataFrame.from_dict(feature_big_dict)

changeset_df.to_csv('/Users/dansvenonius/Desktop/Data/changeset_data.csv')
user_df.to_csv('/Users/dansvenonius/Desktop/Data/user_data.csv')
feature_df.to_csv('/Users/dansvenonius/Desktop/Data/feature_data.csv')
