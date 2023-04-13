from sklearn.model_selection import train_test_split

from Ovid.util import finish_row
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
from Ovid.models.Ovid import OvidEstimator
import Ovid.util as util
from DataAnalysis.HelperFunctions import print_report


class Ovid:
    def __init__(self, no_changeset_features=21, no_user_features=9, no_edit_features=15):
        # detta är initialvärden för my_data.
        # för original_data, dvs för den datan som är närmst det Ovid själva använde, så är det (20, 9, 15).
        # skillnaden är att i my_data har vi imagery_used och nprev_changesets men inte acc_created, men i original_data har vi vice versa.
        self.scaler = None
        self.clf = None

        self.max_edits = 20

        # dimensions
        self.no_changeset_features = no_changeset_features # annars None. Detta är förutom imagery_used
        self.no_user_features = no_user_features # annars None. Denna är förutom nprev_changesets, som vi nog vill ha med. Vi har heller inte med timestamps eller så
        self.no_edit_features = no_edit_features * self.max_edits # för ett element.

    def store(self, path):
        print("Storing model to: " + path)
        self.clf.save(path + "_network")

    def load(self, path):
        self.clf = OvidEstimator(1, 1, 1)
        self.clf.load_model(path + "_network")

    def compute_features(self):

        # ordningen är user features - changeset features - edit features - nåågon weird mask.
        # De parsar lite timestamps och sådant där... Vi kan ju se vad vi väljer att göra på den fronten

        path = '/Users/dansvenonius/PycharmProjects/MastersThesis/Dataset/'

        # börjar med user features:
        # Denna skiljer sig lite med Ovids originalkod: for some reason använder de inte nprev_changesets?
        # Samtidigt säger de inte att de använder den i uppsatsen heller.
        # Vi har inte top_12_tags_used, de har det.
        # Vi använder inte account_created, de gör det. Är dock enkelt att fixa
        user_features = pd.read_csv(path + "user_data.csv").set_index("cs_id")
        user_features = user_features[['create',
                                       'modify',
                                       'delete',
                                       'contributions',
                                       'create_nodes',
                                       'create_ways',
                                       'create_relations',
                                       'nprev_changesets', # de verkar inte använda denna?
                                       'active_weeks',
                                       'acc_created']].rename(columns={'create': 'create_u', 'modify': 'modify_u', 'delete': 'delete_u'})
        user_features['acc_created'] = user_features['acc_created'].apply(util.parse_timestamp_feature)

        # self.no_user_features = user_features.shape[1]

        # changeset features
        changeset_features = pd.read_csv(path + 'changeset_data.csv').set_index("cs_id")

        # Denna är helt identisk med Ovids originalkod, med ett undantag: de använder inte imagery_used här
        # (men säger att de gör det i deras paper? Lite oklart.)
        feature_names = ["create",
                         "modify",
                         "delete",
                         "edits",
                         "nnodes",
                         "nways",
                         "nrelations",
                         "min_lat",
                         "min_lon",
                         "max_lat",
                         "max_lon",
                         "box_size",
                         "comment_len",
                         "imagery_used", # de verkar inte använda denna?
                         "editor_app"]

        changeset_features = changeset_features[feature_names].rename(columns={'create': 'create_cs', 'modify': 'modify_cs', 'delete': 'delete_cs'})

        changeset_features = changeset_features.join(pd.get_dummies(changeset_features["editor_app"]))
        del changeset_features["editor_app"]

        changeset_features = changeset_features.replace([np.inf], 0)

        self.no_changeset_features = changeset_features.shape[1]
        features = user_features.join(changeset_features)

        # vi använder inte kommentarerna
        # comments = pd.read_csv(self.config.comments, sep="\t", index_col=["changeset"])
        # comments["comment"] = comments["comment"].fillna("")
        # comments["comment_length"] = comments["comment"].str.len()
        # comments = comments[["comment_length"]]
        # self.no_changeset_features += comments.shape[1]
        # features = features.join(comments)

        # edit features
        # Denna är helt identisk med Ovids originalkod, med ett undantag: vi har inte geom_dist.
        target_features = ["node",
                           "way",
                           "relation",
                           "create",
                           "modify",
                           "delete",
                           # 'version', denna kommer in på anat håll, men no_edit_features får alltså +1 gpa denna
                           "nprev_auths",
                           "weeks_to_prev",
                           "name_changed",
                           "ntags",
                           "nvalid_tags",
                           "nprev_valid_tags",
                           # "geom_dist", lite oklart vad denna är, men verkar vara något mått på hur noder/ways har flyttats. Skippar for now. Verkar vara att vi behöver mer information för denna, typ lat/lon. Då skippar vi
                           "tags_deleted",
                           "tags_added"]

        edits_df = pd.read_csv(path + 'element_data.csv', index_col=["cs_id", "type", "element_id", "version"])

        operation_dummies = pd.get_dummies(edits_df["operation"])
        edits_df = edits_df.join(operation_dummies)

        dummies = pd.get_dummies(edits_df.index.get_level_values(1))
        dummies = dummies.set_index(edits_df.index)

        edits_df = edits_df.join(dummies)
        edits_df = edits_df[target_features]
        edits_df = edits_df.sort_index()

        current_changeset = -1
        edit_count = -1
        rows = []
        current_row = []
        current_mask_row = []
        for r in tqdm(edits_df.iterrows(), total=len(edits_df.index), desc="Computing edit features"):
            changeset = r[0][0]

            if changeset != current_changeset:
                if current_changeset != -1:
                    rows.append(
                        finish_row(current_row, edit_count, self.max_edits, len(target_features) + 1, current_mask_row))

                current_changeset = changeset
                edit_count = 0
                current_row = [current_changeset]
                current_mask_row = []

            # version
            current_row.append(r[0][3])

            # other features
            for current_feature in target_features:
                current_row.append(r[1][current_feature])

            current_mask_row.append(True)

            edit_count += 1
        rows.append(finish_row(current_row, edit_count, self.max_edits, len(target_features) + 1, current_mask_row))

        edit_cols = ["changeset"]
        for i in range(self.max_edits):
            for tf in ["version"] + target_features:
                edit_cols.append(str(i) + "_" + tf)

        for i in range(self.max_edits):
            edit_cols.append("mask_" + str(i))

        edit_features = pd.DataFrame(rows, columns=edit_cols).set_index("changeset")
        self.no_edit_features = edit_features.shape[1] - self.max_edits

        features = features.join(edit_features, how="left").fillna(0)
        features.replace(['True', 'False', '0'], [True, False, False], inplace=True)

        return features

    def _get_input_parts(self, X):
        changeset_features_end = self.no_user_features + self.no_changeset_features
        edit_features_end = changeset_features_end + self.no_edit_features

        user_features = X[:, 0:self.no_user_features]
        changeset_features = X[:, self.no_user_features:changeset_features_end]

        edit_features = X[:, changeset_features_end:edit_features_end]

        return [changeset_features, user_features, edit_features]

    def get_edit_mask(self, X):
        edit_features_end = self.no_user_features + self.no_changeset_features + self.no_edit_features
        edit_mask_end = edit_features_end + self.max_edits
        edit_mask = X[:, edit_features_end:edit_mask_end]
        return edit_mask.astype('uint8')

    def fit(self, X, y, X_val, y_val):
        edit_mask = self.get_edit_mask(X)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        val_edit_mask = self.get_edit_mask(X_val)
        X_val_scale = self.scaler.transform(X_val)

        val_data = (self._get_input_parts(X_val_scale) + [val_edit_mask], y_val)

        self.clf = OvidEstimator(no_changeset_features=self.no_changeset_features,
                                 no_user_features=self.no_user_features,
                                 no_edit_features=self.no_edit_features,
                                 max_edits=self.max_edits)

        self.clf.build_model()
        self.clf.fit(self._get_input_parts(X) + [edit_mask], y, val_data)

    def predict(self, X):
        edit_mask = self.get_edit_mask(X)
        X = self.scaler.transform(X)
        result = self.clf.predict(self._get_input_parts(X) + [edit_mask])
        result = result >= 0.5
        return result.reshape(-1, )

    def predict_proba(self, X):
        edit_mask = self.get_edit_mask(X)
        X = self.scaler.transform(X)
        result = self.clf.predict(self._get_input_parts(X) + [edit_mask])
        return result.reshape(-1, )

    def fit_scaler(self, X):
        self.scaler = StandardScaler()
        self.scaler.fit_transform(X)

def run_ovid():
    # to train a model a few hardcoded things need to be taken into consideration. Path where we save the model and what we have in features_df.
    # if ovid_data.csv does not contain the necessary data, run compute_features() again with the appropriate changes (also hardcoded)
    ovid = Ovid()

    original_data = False
    my_data = True

    labels_df = pd.read_csv('/Users/dansvenonius/PycharmProjects/MastersThesis/Dataset/labels.csv').set_index("cs_id")
    features_df = pd.read_csv('/Users/dansvenonius/PycharmProjects/MastersThesis/Dataset/ovid_data.csv').set_index("cs_id")
    if original_data:
        features_df.drop(labels=['nprev_changesets', 'imagery_used'], axis='columns', inplace=True)
    elif my_data:
        features_df.drop(labels='acc_created', axis='columns', inplace=True)

    labels_features = labels_df.join(features_df, how="left").reset_index()
    labels_features = labels_features.fillna(0)

    y = labels_features["label"].to_numpy()
    X = labels_features.drop(["label", "cs_id"], axis=1).to_numpy()

    # 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 10% validation, 70% train
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    ovid.fit(X_train, y_train, X_val, y_val)
    ovid.store("./MastersThesis/models/ovid_model_my_data")

    pred_test = ovid.predict(X_test)
    pred_train = ovid.predict(X_train)
    print_report(pred_train, y_train, pred_test, y_test, title="Ovid Classification")

