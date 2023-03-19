import pandas as pd

# data = 'node_create_data'
#
# df_10k = pd.read_csv('/Users/dansvenonius/Desktop/Preprocessing output/' + data + '.csv').iloc[:, 1:]
# df_17k = pd.read_csv('/Users/dansvenonius/Desktop/Preprocessing output/' + data + '_17k.csv').iloc[:, 1:]
# df_total = pd.concat([df_10k, df_17k], ignore_index=True).drop_duplicates()
# df_total.to_csv('/Users/dansvenonius/Desktop/Preprocessing output/' + data + '_total.csv')

# d1 = {0: {'one': 'ett', 'two': 'två'}, 1: {'three': 'tre', 'four': 'fyra'}}
# for idx, nested in d1.items():
#     if idx == 0:
#         nested['one'] = 'ETT'
#     else:
#         nested['three'] = 'TRE'
#
# print(d1)


# ids_pre_relations = pd.read_csv('/Users/dansvenonius/Desktop/Preprocessing output/relation_data.csv').loc[:, 'element_id']
# ids_proc_relations = pd.read_csv('/Users/dansvenonius/Desktop/History output/elements_modifydelete_data.csv').loc[:, 'id']
# bad_ids = pd.read_csv('/Users/dansvenonius/Desktop/History output/bad_elements_data.csv').loc[:, 'element_id']
#
# count = 0
# for index, id in ids_pre_relations.items():
#     if id in bad_ids.values:
#         count += 1
#
# sorted_bad = sorted(list(bad_ids))
#
#
# pre_set = set(ids_pre_relations.values)
# proc_set = set(ids_proc_relations.values)
#
# sorted_pre = sorted(list(ids_pre_relations))
# sorted_proc = sorted(list(ids_proc_relations))
#
# diff_set = pre_set.difference(proc_set)
# print(diff_set)