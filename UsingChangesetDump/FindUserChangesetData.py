import osmium
import time
import pandas as pd
from UsingMainAPI import MainAPIHelperFunctions as mainhf


def init_user_data(user_df):
    big_dict = {}
    for index, row in user_df.iterrows():
        nested_user_dict = {'uid': row['uid'],
                            'cs_id': row['cs_id'],
                            'prev_cs_times': [],
                            'nprev_cs': 0,
                            'cs_created_at': mainhf.string_to_datetime(row['cs_created_at'])}
        if index % 10000 == 0:
            print(index)
        if row['uid'] in big_dict.keys():
            big_dict[row['uid']][row['cs_id']] = nested_user_dict
        else:
            big_dict[row['uid']] = {row['cs_id']: nested_user_dict}
    return big_dict



def get_active_weeks(user_dict):
    times = sorted(user_dict['prev_cs_times'])
    if len(times) == 0:
        return 0
    nactive_weeks = 1
    t1 = times[0]
    for i in range(len(times) - 1):
        t2 = times[i + 1]
        if (t2 - t1).days >= 7:
            nactive_weeks += 1
            t1 = t2
    return nactive_weeks


class ChangesetHistoryHandler(osmium.SimpleHandler):
    def __init__(self, user_df):
        osmium.SimpleHandler.__init__(self)
        self.user_dict = init_user_data(user_df)

    def changeset(self, cs):
        if cs.uid in self.user_dict.keys():
            for cs_dict in self.user_dict[cs.uid].values():
                if cs.closed_at < cs_dict['cs_created_at']:
                    cs_dict['prev_cs_times'].append(cs.closed_at)
                    cs_dict['nprev_cs'] += 1


########### INPUT ############
reverted_data = True
##############################

input_file = '/Users/dansvenonius/Desktop/changesets-230227.osm.bz2'
if reverted_data:
    csv_file = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/user_data.csv'
else:
    csv_file = '/Users/dansvenonius/Desktop/Preprocessing output/Not Reverted Data/user_data.csv'


tstart = time.perf_counter()
user_df = pd.read_csv(csv_file)

chh = ChangesetHistoryHandler(user_df)
chh.apply_file(input_file)
#chh.apply_file("/Users/dansvenonius/Desktop/test_changeset.xml")

big_user_dict = {}
for uid_to_cs in chh.user_dict.values():
    for cs_to_data in uid_to_cs.values():
        big_user_dict = mainhf.add_dict(big_user_dict, {'cs_id': cs_to_data['cs_id'],
                                                        'nprev_changesets': cs_to_data['nprev_cs'],
                                                        'active_weeks': get_active_weeks(cs_to_data)})

if 'active_weeks' in user_df.columns:
    del user_df['active_weeks']
if 'nprev_changesets' in user_df.columns:
    del user_df['nprev_changesets']

user_df = user_df.merge(pd.DataFrame.from_dict(big_user_dict), how='inner', on='cs_id')
user_df.to_csv(csv_file, index=False)

print(time.perf_counter()-tstart)
