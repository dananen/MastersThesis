from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from DataAnalysis import HelperFunctions as hf, TweakedOvid
import numpy as np
import pandas as pd
import time


random_state = 42
dataset_path = '/Users/dansvenonius/PycharmProjects/MastersThesis/Dataset/'

MLP_grid_1 = {
    'hidden_layer_sizes': [10, (10, 10), (10, 10, 10), 100, (100, 100), (100, 100, 100), 1000, (1000, 1000), (1000, 1000, 1000)],
    'max_iter': [200, 400, 600]
}
result_MLP_grid_1 = {
    'hidden_layer_sizes': (100, 100),
    'max_iter': 400
}

MLP_grid_2 = {
    'alpha': [1e-3, 1e-4, 1e-5],
    'batch_size': [100, 200, 300],
    'learning_rate_init': [1e-2, 1e-3, 1e-4],
}

rf_grid = {
    'n_estimators': [10, 25, 50, 100, 250, 500],
    'max_depth': [5, 10, 25, 50],
    'max_features': [2, 5, 10],
    #'min_samples_leaf': [3, 10]
}

cb_grid = {
    'iterations': [10, 25, 50, 100, 250, 500],
    'depth': [4, 7, 10],
    'l2_leaf_reg': [0.5, 3, 10],
    # 'learning_rate': [0.005, 0.03, 0.1],
}

X, y = hf.get_prepped_csudf(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), batch_size=200, alpha=1e-5, max_iter=400, learning_rate_init=0.001)
mlp.fit(X_train_scaled, y_train)
pred = mlp.predict(X_test_scaled)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=2))

pred = mlp.predict(X_train_scaled)
print(confusion_matrix(y_train, pred))
print(classification_report(y_train, pred, digits=2))



# rf = RandomForestClassifier(**{'max_depth': 50, 'max_features': 5, 'n_estimators': 500})
# #rf = RandomForestClassifier(**{'max_depth': 50, 'max_features': 5, 'n_estimators': 500})
# rf.fit(X_train, y_train)
# print(classification_report(y_train, rf.predict(X_train)))
# print(classification_report(y_test, rf.predict(X_test)))
# print(confusion_matrix(y_train, rf.predict(X_train)))
# print(confusion_matrix(y_test, rf.predict(X_test)))
# # hf.rfc_importances(X_train, y_train, rf)
# idx = np.flip(np.argsort(rf.feature_importances_))
# print(rf.feature_importances_[idx])
# print(rf.feature_names_in_[idx])


# gs = GridSearchCV(rf, rf_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=3)
# gs.fit(X_train, y_train)

# göra om grid_searchen för MLP? Den är brutalt kass, orimligt kass verkligen. Stämmer inte alls med vad vi hade för resultat under grid_searchen
# bästa outputten från RF är {'max_depth': 50, 'max_features': 5, 'n_estimators': 500} med acc 0.87368, men en betydligt mindre modell är
# {'max_depth': 25, 'max_features': 5, 'n_estimators': 250} med acc 0.87317. Men egentligen är det inga problem att köra den stora, annat
# än att den potentiellt blir jävligt övertränad. Vi kan kolla lite. Båda blir 100% övertränade

# bästa outputten från Grd search 1 är (1000, 1000) max_iter = 600, men väldigt nära är (100, 100), max_iter = 400.
# köra en liten fuling och säga att (100, 100) max_iter = 400 är bäst? Jag tycker typ det... Sparar stört mycket tid och ingen vet. Så gör vi
# {'alpha': 1e-05, 'batch_size': 200, 'learning_rate_init': 0.001} från omgång 2

# print(gs.cv_results_)
# print(gs.cv_results_['mean_test_score'])
# print(gs.cv_results_['params'])
# res_gs1 = gs.cv_results_['params'][np.argmax(gs.cv_results_['mean_test_score'])]
# print(res_gs1)


# t1 = time.perf_counter()
# mlp = MLPClassifier(**{'alpha': 1e-05, 'batch_size': 200, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (1000, 1000), 'max_iter': 600})
# mlp.fit(X_train, y_train)
# print(classification_report(y_train, mlp.predict(X_train)))
# print(confusion_matrix(y_train, mlp.predict(X_train)))
# print('-----------------------------------------')
# print(classification_report(y_test, mlp.predict(X_test)))
# print(confusion_matrix(y_test, mlp.predict(X_test)))
# print(time.perf_counter() - t1)

# gs = GridSearchCV(mlp, MLP_grid_2, scoring='accuracy', n_jobs=-1, cv=3, verbose=3)
# gs.fit(X_train_scaled, y_train)

# print(gs.cv_results_)
# print(gs.cv_results_['mean_test_score'])
# print(gs.cv_results_['params'])
# idx = np.flip(np.argsort(gs.cv_results_['mean_test_score']))
# print(idx)
#
# for i in idx:
#     print(gs.cv_results_['mean_test_score'][i])
#     print(gs.cv_results_['params'][i])


# {'depth': 10, 'iterations': 500, 'l2_leaf_reg': 0.5} för catboost
# t1 = time.perf_counter()
# cb = CatBoostClassifier(logging_level='Silent', **{'depth': 10, 'iterations': 500, 'l2_leaf_reg': 0.5})
# cb.fit(X_train, y_train)
# cb_pred_train = cb.predict(X_train) == 'True'
# cb_pred_test = cb.predict(X_test) == 'True'
# print(classification_report(y_train, cb_pred_train))
# print(classification_report(y_test, cb_pred_test))
# print(confusion_matrix(y_train, cb_pred_train))
# print(confusion_matrix(y_test, cb_pred_test))

# res = cb.grid_search(cb_grid, X=X_train, y=y_train, cv=3, partition_random_seed=random_state, verbose=10)
# print(res['cv_results'])
# print(res['params'])
# print(time.perf_counter() - t1)

# ovid = TweakedOvid.Ovid()
# features_df = pd.read_csv('/Users/dansvenonius/PycharmProjects/MastersThesis/Dataset/ovid_data.csv').set_index("cs_id")
# labels = pd.read_csv('/Users/dansvenonius/PycharmProjects/MastersThesis/Dataset/labels.csv').set_index('cs_id')
# labels_features = labels.join(features_df, how="left").reset_index()
# y_ovid = labels_features["label"].to_numpy()
# X_ovid = labels_features.drop(["label", "cs_id"], axis='columns').to_numpy()
#
# X_train_ovid, X_test_ovid, y_train_ovid, y_test_ovid = train_test_split(X_ovid, y_ovid, test_size=0.2, random_state=random_state)
# X_train_ovid, X_val_ovid, y_train_ovid, y_val_ovid = train_test_split(X_train_ovid, y_train_ovid, test_size=0.25, random_state=random_state)
# ovid.fit_scaler(X_train_ovid)
# ovid.fit(X_train_ovid, y_train_ovid, X_val_ovid, y_val_ovid, epochs=100)
#
# ovid_pred_test = ovid.predict(X_test_ovid)
# ovid_pred_train = ovid.predict(X_train_ovid)
# print(classification_report(y_train_ovid, ovid_pred_train))
# print(confusion_matrix(y_train_ovid, ovid_pred_train))
# print(classification_report(y_test_ovid, ovid_pred_test))
# print(confusion_matrix(y_test_ovid, ovid_pred_test))




