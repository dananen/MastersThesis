import osmium as osm
import pandas as pd
import time
import numpy as np


def find_samples():
    reverted_cs_ids = pd.read_csv('/Users/dansvenonius/Desktop/Misc output/Reverted Data/reverted_changesets.csv').loc[:, 'reverted_id']
    ids_to_sample = set()
    while len(ids_to_sample) < reverted_cs_ids.size:
        sample = np.random.randint(10000000, 133064137+1) # we look at cs_ids with 8 or 9 digits for "revert" comments. This way we scan through the same.
        if sample not in reverted_cs_ids.values:
            ids_to_sample.add(sample)
    return ids_to_sample


class ChangesetHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.elements = []
        self.ids_to_sample = find_samples()

    def changeset(self, cs):
        if cs.id in self.ids_to_sample:
            self.elements.append([cs.id, cs.created_at])


t1 = time.perf_counter()
file = '/Users/dansvenonius/Desktop/changesets-230227.osm.bz2'

cs_handler = ChangesetHandler()
cs_handler.apply_file(file)

elements_df = pd.DataFrame(cs_handler.elements, columns=['cs_id', 'created_at'])
elements_df.to_csv('/Users/dansvenonius/Desktop/Misc output/Not Reverted Data/false_label_changesets.csv', index=False)
print('Time to run code: {0} seconds'.format(round(time.perf_counter()-t1)))

