import osmium as osm
import pandas as pd
import time
from UsingMainAPI import MainAPIHelperFunctions as mainhf


class ChangesetKeywordFilter(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.revert_elements = []
        #self.vandalism_elements = []

    def changeset(self, cs):
        comment = cs.tags.get("comment")
        if type(comment) is type(None):
            return

        lower_comment = comment.lower()
        ids = mainhf.containsID(lower_comment)
        for reverted_id in ids:
         #   if "vandalism" in lower_comment:
         #       self.vandalism_elements.append([cs.id, cs.created_at, comment, reverted_id])
            if "revert" in lower_comment and "vandalism" not in lower_comment:
                self.revert_elements.append([cs.id, cs.created_at, comment, reverted_id])


class ChangesetTimestampHandler(osm.SimpleHandler):
    def __init__(self, ids):
        osm.SimpleHandler.__init__(self)
        self.cs_id_timestamp = []
        self.reverted_ids = {id for id in ids}

    def changeset(self, cs):
        if cs.id in self.reverted_ids:
            self.cs_id_timestamp.append([cs.id, cs.created_at])


t1 = time.perf_counter()
file = '/Users/dansvenonius/Desktop/changesets-230227.osm.bz2'
#file = '/Users/dansvenonius/Desktop/test_changeset.xml'

keyword_cs_handler = ChangesetKeywordFilter()
keyword_cs_handler.apply_file(file)

print('Time to run first iteration: {0} seconds'.format(round(time.perf_counter()-t1)))
t2 = time.perf_counter()

colnames = ['revert_cs_id', 'revert_created_at', 'comment', 'reverted_id']
#vandalism_elements = pd.DataFrame(keyword_cs_handler.vandalism_elements, columns=colnames)
reverted_elements = pd.DataFrame(keyword_cs_handler.revert_elements, columns=colnames)

reverted_cs_handler = ChangesetTimestampHandler(reverted_elements.loc[:, 'reverted_id'].values)
reverted_cs_handler.apply_file(file)
reverted_elements_w_timestamps = reverted_elements.merge(
    pd.DataFrame(reverted_cs_handler.cs_id_timestamp, columns=['reverted_id', 'reverted_created_at']),
    how='inner', on='reverted_id')

reverted_elements_w_timestamps.to_csv('/Users/dansvenonius/Desktop/Misc output/reverted_changesets.csv', index=False)

print('Time to run second iteration: {0} seconds'.format(round(time.perf_counter()-t2)))
print('Time to run second code: {0} seconds'.format(round(time.perf_counter()-t1)))
