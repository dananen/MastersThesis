import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import os

import sklearn.model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier


def split_classes(data_df, labels, with_label=False):
    # returning true first, false second
    data_w_labels = data_df.merge(labels, on='cs_id')
    rev_data, non_rev_data = data_w_labels.loc[data_w_labels['label'] == True], data_w_labels.loc[data_w_labels['label'] == False]
    if not with_label:
        del rev_data['label']
        del non_rev_data['label']
    return rev_data, non_rev_data


def rev_nonrev_histograms(data_df, labels, data_point, lower, upper, show_plot=True, save_data=False, save_folder='', title='', ltitle='',
        rtitle='', step=1.0, sharey=True, xlog=True, ylog=False, categorical=False):

    rev_data, non_rev_data = split_classes(data_df, labels)
    if sharey:
        fig, axes = plt.subplots(1, 2, sharey='row')
    else:
        fig, axes = plt.subplots(1, 2)
    axes[0].set_xlabel(data_point + ' in changeset')
    axes[1].set_xlabel(data_point + ' in changeset')
    axes[0].set_ylabel('count')
    axes[1].set_ylabel('count')
    if len(title) > 0:
        fig.suptitle(title)
    if ylog:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
    if categorical:
        rev_data[data_point].value_counts().plot(kind='bar', ax=axes[0])
        non_rev_data[data_point].value_counts().plot(kind='bar', ax=axes[1])
    else:
        bins = np.arange(lower, upper, step=step, dtype=float)
        if xlog:
            axes[0].set_xscale('log')
            axes[1].set_xscale('log')
            bins = 10**bins
            if lower >= 0:
                np.insert(bins, 0, 0)
        rev_data.hist(data_point, ax=axes[0], bins=bins)
        non_rev_data.hist(data_point, ax=axes[1], bins=bins)

    axes[0].title.set_text(ltitle)
    axes[1].title.set_text(rtitle)
    if save_data and save_folder:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        plt.savefig(save_folder + '/' + data_point + '.png')
    if show_plot:
        plt.show()
    plt.close()


def get_histogram_variable_settings(lower, upper, step, sharey, xlog, ylog, catgegorical):
    return {'lower': lower, 'upper': upper, 'step': step, 'sharey': sharey, 'xlog': xlog, 'ylog': ylog, 'categorical': catgegorical}


def get_histogram_settings(cs_data=None, u_data=None, el_data=None, include_cs=True, include_u=True, include_el=True, var=None,
                           cs_folder='changeset data histograms', u_folder='user data histograms', el_folder='element data histograms'):
    d = {}
    if include_cs:
        d['changeset data'] = {
            'title': 'Changeset Data',
            'data': cs_data if cs_data else pd.read_csv('Dataset/changeset_data.csv'),
            'save_path': '/Users/dansvenonius/Desktop/Screenshots/' + cs_folder,
            'create': get_histogram_variable_settings(0, 6, 0.5, True, True, False, False),
            'modify': get_histogram_variable_settings(0, 6, 0.5, True, True, False, False),
            'delete': get_histogram_variable_settings(0, 6, 0.5, True, True, False, False),
            'edits': get_histogram_variable_settings(0, 6, 0.5, True, True, False, False),
            'nnodes': get_histogram_variable_settings(0, 6, 0.5, True, True, False, False),
            'nways': get_histogram_variable_settings(0, 5, 0.5, True, True, False, False),
            'nrelations': get_histogram_variable_settings(0, 3, 0.5, True, True, False, False),
            'min_lon': get_histogram_variable_settings(-120, 120, 30, True, False, False, False),
            'max_lon': get_histogram_variable_settings(-120, 120, 30, True, False, False, False),
            'min_lat': get_histogram_variable_settings(-120, 120, 30, True, False, False, False),
            'max_lat': get_histogram_variable_settings(-120, 120, 30, True, False, False, False),
            'box_size': get_histogram_variable_settings(-13, 8, 1, True, True, False, False),
            'comment_len': get_histogram_variable_settings(0, 4, 1, True, True, False, False),
            'imagery_used': get_histogram_variable_settings(-1, -1, -1, True, False, False, True),
            'editor_app': get_histogram_variable_settings(-1, -1, -1, True, False, False, True)
        }
    if include_u:
        d['user data'] = {
            'title': 'User Data',
            'data': u_data if u_data else pd.read_csv('Dataset/user_data.csv'),
            'save_path': '/Users/dansvenonius/Desktop/Screenshots/' + u_folder,
            'create': get_histogram_variable_settings(0, 9, 1, True, True, False, False),
            'modify': get_histogram_variable_settings(0, 9, 1, True, True, False, False),
            'delete': get_histogram_variable_settings(0, 9, 1, True, True, False, False),
            'contributions': get_histogram_variable_settings(0, 9, 1, True, True, False, False),
            'create_nodes': get_histogram_variable_settings(0, 9, 1, True, True, False, False),
            'create_ways': get_histogram_variable_settings(0, 9, 1, True, True, False, False),
            'create_relations': get_histogram_variable_settings(0, 6, 1, True, True, False, False),
            'active_weeks': get_histogram_variable_settings(0, 4, 0.5, True, True, False, False),
            'nprev_changesets': get_histogram_variable_settings(0, 7, 1, True, True, False, False)
        }
    if include_el:
        d['element data'] = {
            'title': 'Element Data',
            'data': el_data if el_data else pd.read_csv('Dataset/element_data.csv'),
            'save_path': '/Users/dansvenonius/Desktop/Screenshots/' + el_folder,
            'type': get_histogram_variable_settings(-1, -1, -1, False, False, True, True),
            'operation': get_histogram_variable_settings(-1, -1, -1, False, False, True, True),
            'name_changed': get_histogram_variable_settings(-1, -1, -1, False, False, True, True),
            'version': get_histogram_variable_settings(0, 4, 0.25, False, True, True, False),
            'weeks_to_prev': get_histogram_variable_settings(0, 4, 0.25, False, True, False, False),
            'nprev_auths': get_histogram_variable_settings(0, 250, 10, False, False, True, False),
            'ntags': get_histogram_variable_settings(0, 600, 25, False, False, True, False),
            'tags_added': get_histogram_variable_settings(0, 250, 3, False, False, True, False), # olika för andra histogrammet. FIXA
            'tags_deleted': get_histogram_variable_settings(0, 400, 3, False, False, True, False), # olika för andra histogrammet. FIXA
            'nvalid_tags': get_histogram_variable_settings(0, 12, 1, False, False, True, False),
            'nprev_valid_tags': get_histogram_variable_settings(0, 12, 1, False, False, True, False)
        }

    if var:
        if len(d) != 1:
            raise Exception("var can only be initialized if exactly one of cs_data, u_data, el_data is True")
        data = next(iter(d))
        temp = {data: {}}
        for key, value in d.items():
            if key in var:
                temp[data][key] = value
            else:
                raise Exception('var contains bad variable ' + key)
        return temp
    return d


def make_histograms(save_data, ltitle, rtitle, cs_data=None, u_data=None, el_data=None, show_plot=True, include_cs=True, include_u=True, include_el=True, var=None,
                    cs_folder='changeset data histograms', u_folder='user data histograms', el_folder='element data histograms'):

    settings = get_histogram_settings(cs_data=cs_data, u_data=u_data, el_data=el_data,include_cs=include_cs, include_u=include_u,
                                      include_el=include_el, var=var, cs_folder=cs_folder, u_folder=u_folder, el_folder=el_folder)
    labels = pd.read_csv('Dataset/labels.csv')
    for data_category in settings.values():
        for variable, setting in data_category.items():
            if variable not in ['title', 'data', 'save_path']:
                rev_nonrev_histograms(data_category['data'], labels, variable, setting['lower'], setting['upper'], show_plot=show_plot, save_data=save_data,
                           save_folder=data_category['save_path'], title=data_category['title'], ltitle=ltitle+variable,
                           rtitle=rtitle+variable, step=setting['step'], sharey=setting['sharey'], xlog=setting['xlog'],
                           ylog=setting['ylog'], categorical=setting['categorical'])



def get_prepped_csudf(path, to_numpy=False, multi_label=False, multi_file=None):
    if (multi_label and multi_file is None) or (not multi_label and multi_file is not None):
        raise Exception("either none or both of multi_label and multi_file must be used.")

    csdf = pd.read_csv(path + 'prepped_changeset_data.csv').set_index("cs_id")
    csdf.drop(labels=['created_at'], axis='columns', inplace=True)

    udf = pd.read_csv(path + "user_data.csv").set_index("cs_id")
    udf.drop(labels=['cs_created_at', 'uid', 'acc_created'], axis='columns', inplace=True)

    if multi_label:
        labels = pd.read_csv(path + multi_file).set_index('cs_id')
        label_names = ['full_revert', 'partial_revert', 'non_revert']
    else:
        labels = pd.read_csv(path + 'labels.csv').set_index('cs_id')
        label_names = ['label']
    features = csdf.join(udf, how='left')
    labels_features = labels.join(features, how='left')
    if not to_numpy:
        return labels_features.drop(labels=label_names, axis='columns'), labels_features.loc[:, label_names]
    else:
        labels_features.reset_index(inplace=True)
        y = labels_features[label_names].to_numpy()
        if multi_label:
            y = y.argmax(axis=1)
        return labels_features.drop(labels=label_names + ["cs_id"], axis='columns').to_numpy(), y


def prep_df(df, type, keep_csids=False, trim_only_csids=False, to_numpy=False, one_hot=True, exclude_osmgo=True):

    if df is None:
        return None

    if type.lower() == "labels":
        to_trim = []
        if not keep_csids:
            to_trim.append('cs_id')
        binary_cat = ['label']
        categoricals = ['label']
    elif type.lower() == "changeset":
        to_trim = ['uid', 'created_at']
        if not keep_csids:
            to_trim.append('cs_id')
        binary_cat = ['imagery_used']
        categoricals = ['imagery_used', 'editor_app']
    elif type.lower() == "user":
        to_trim = ['uid', 'cs_created_at', 'acc_created']
        if not keep_csids:
            to_trim.append('cs_id')
        binary_cat = []
        categoricals = []
    elif type.lower() == "element":
        to_trim = ['element_id']
        if not keep_csids:
            to_trim.append('cs_id')
        binary_cat = ['name_changed']
        categoricals = ['name_changed', 'operation', 'type']
    else:
        raise Exception("Type has to be 'changeset', 'user', 'element' or 'labels'.")

    prepped = df.copy()
    if type == 'changeset' and exclude_osmgo:
        prepped.loc[:, 'editor_app'].replace('osm go!', 'other', inplace=True)
    if trim_only_csids:
        del prepped['cs_id']
    else:
        prepped = prepped.loc[:, ~prepped.columns.isin(to_trim)]
        for var in categoricals:
            if var in binary_cat:
                prepped.loc[:, var] = np.where(prepped.loc[:, var] == True, 1, 0)
            elif one_hot:
                encoder = OneHotEncoder()
                encoded_df = pd.DataFrame(encoder.fit_transform(prepped[[var]]).toarray())
                encoded_df.index = prepped.index
                encoded_df.columns = encoder.get_feature_names_out()
                prepped = pd.concat([prepped, encoded_df], axis=1)
                del prepped[var]

    if to_numpy:
        prepped = prepped.to_numpy()
        if type == 'labels':
            prepped = prepped.ravel()
    return prepped


def tSNEplot(labels, csdf=None, udf=None, eldf=None):
    if eldf is not None:
        raise Exception("No support for eldf yet.")
    if csdf is None and udf is None and eldf is None:
        raise Exception("Need at least one of csdf, udf or eldf (not now) to perform analysis.")

    title = 't-SNE for '
    data = []
    if csdf is not None:
        rev_csdf, nonrev_csdf = split_classes(csdf, labels)
        data.append((prep_df(rev_csdf, 'changeset', to_numpy=True), prep_df(nonrev_csdf, 'changeset', to_numpy=True)))
        title += 'changeset + '
    if udf is not None:
        rev_udf, nonrev_udf = split_classes(udf, labels)
        data.append((prep_df(rev_udf, 'user', to_numpy=True), prep_df(nonrev_udf, 'user', to_numpy=True)))
        title += 'user'
    if eldf is not None:
        ### TODO. No support for this yet
        #rev_eldf, nonrev_eldf = split_classes(eldf, labels)
        pass

    for i, (rev, nonrev) in enumerate(data):
        if i == 0:
            rev_data, nonrev_data = rev, nonrev
        else:
            rev_data, nonrev_data = np.hstack((rev_data, rev)), np.hstack((nonrev_data, nonrev))

    tsne = TSNE()
    rev_tsne, nonrev_tsne = tsne.fit_transform(rev_data), tsne.fit_transform(nonrev_data)
    plt.scatter(rev_tsne[:, 0], rev_tsne[:, 1], color='red', label='reverted', alpha=0.5)
    plt.scatter(nonrev_tsne[:, 0], nonrev_tsne[:, 1], color='blue', label='not reverted', alpha=0.5)
    plt.legend()
    plt.title(title + " data")
    plt.show()


def get_feature_names(cs=True, u=True, el=False):
    names = []
    if cs:
        for feature in ['create', 'modify', 'delete', 'edits', 'nnodes', 'nways', 'nrelations',
                        'min_lon', 'max_lon', 'min_lat', 'max_lat', 'box_size', 'comment_len',
                        'imagery_used', 'editor_app_go map!!', 'editor_app_josm',  # 'editor_app_osm go!',
                        'editor_app_other', 'editor_app_potlatch', 'editor_app_streetcomplete', 'editor_app_vespucci']:
            names.append(feature)
    if u:
        for feature in ['create', 'modify', 'delete', 'contributions', 'create_nodes', 'create_ways', 'create_relations',
                        'nprev_changesets', 'active_weeks']:
            names.append(feature)
    if el:
        for feature in ['version', 'nprev_auths', 'weeks_to_prev', 'name_changed', 'ntags', 'tags_added',
                        'tags_deleted', 'nvalid_tags', 'nprev_valid_tags', 'operation_create',
                        'operation_delete', 'operation_modify', 'type_node', 'type_relation', 'type_way']:
            names.append(feature)
    return names


def random_forest(y_train, y_test, X_train, X_test, report=True, model=None, shapley=False):
    if model is None:
        model = RandomForestClassifier(n_estimators=25,
                                       max_depth=100,
                                       max_features=15,
                                       max_leaf_nodes=None,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       bootstrap=False)
    model.fit(X_train, y_train.ravel())
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if report:
        print_report(pred_train, y_train, pred_test, y_test,  title="Random Forest Classification")
    if shapley:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)
    return model


def rfc_importances(X_test, y_test, model, cs=True, u=True, el=False):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    forest_importances = pd.Series(importances)
    fig, ax = plt.subplots(2, 1, sharex='all')
    forest_importances.plot.bar(yerr=std, ax=ax[0])
    ax[0].set_title("Feature importances using MDI")
    ax[0].set_ylabel("Mean imp. decrease")
    ax[0].set_title("Mean decrease in impurity")

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=get_feature_names(cs=cs, u=u, el=el))

    forest_importances.plot.bar(yerr=result.importances_std, ax=ax[1])
    ax[1].set_title("Feature importances using permutation on full model")
    ax[1].set_ylabel("Mean acc. decrease")
    ax[1].set_title("Permutation importance")
    fig.tight_layout()
    fig.show()

def mlp_classification(y_train, y_test, X_train, X_test, report=True, model=None, shapley=False):
    if model is None:
        model = MLPClassifier(alpha=0.001,
                              batch_size=200,
                              early_stopping=True,
                              hidden_layer_sizes=1000,
                              learning_rate_init=0.0001,
                              max_iter=400,
                              validation_fraction=0.3)
    scaler = StandardScaler()
    X_train_mlp = scaler.fit_transform(X_train)
    model.fit(X_train_mlp, y_train)

    X_test_mlp = scaler.transform(X_test)
    pred_test = model.predict(X_test_mlp)
    pred_train = model.predict(X_train_mlp)

    if report:
        print_report(pred_train, y_train, pred_test, y_test, title="MLP Classification")
    if shapley:
        explainer = shap.Explainer(lambda X: model.predict([X[:, i] for i in range(X.shape[1])]).flatten())
        shap_values = explainer(X_train_mlp)
        shap.summary_plot(shap_values, X_train_mlp)
    return model

def print_report(pred_train, y_train, pred_test, y_test, target_names=None, title="", normalize=True):
    norm = 'all' if normalize else None

    fig, ax = plt.subplots(1, 2, sharey='row')
    print('######################## TRAINING DATA ###########################')
    if target_names is not None:
        print(classification_report(y_train, pred_train, target_names=target_names))
        cmd = ConfusionMatrixDisplay(confusion_matrix(y_train, pred_train, normalize=norm), display_labels=target_names).plot(ax=ax[0], cmap='Greens')
    else:
        print(classification_report(y_train, pred_train))
        cmd = ConfusionMatrixDisplay(confusion_matrix(y_train, pred_train, normalize=norm)).plot(ax=ax[0], cmap='Greens')
    cmd.im_.colorbar.remove()
    cmd.ax_.set_xlabel('')
    ax[0].set_title('Training data')
    print('########################## TEST DATA #############################')
    if target_names is not None:
        print(classification_report(y_test, pred_test, target_names=target_names))
        cmd = ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test, normalize=norm), display_labels=target_names).plot(ax=ax[1], cmap='Greens')
    else:
        print(classification_report(y_test, pred_test))
        cmd = ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test, normalize=norm)).plot(ax=ax[1], cmap='Greens')
    cmd.im_.colorbar.remove()
    cmd.ax_.set_xlabel('')
    cmd.ax_.set_ylabel('')
    ax[1].set_title('Test data')
    fig.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()

def grid_search_cv(model, param_grid, X_train, y_train, X_test, y_test, scoring='accuracy', n_jobs=None):
    gscv = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=n_jobs, verbose=10, refit=True)
    gscv.fit(X_train, y_train)
    print(gscv.best_score_)
    print(gscv.cv_results_)
    print(gscv.best_params_)
    best_model = gscv.best_estimator_
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds, labels=["Not reverted", "Reverted"]))


def get_2_of_3_classes(path, c1, c2, multi_file, to_numpy=False):
    poss_labels = ['full_revert', 'partial_revert', 'non_revert']
    if c1 not in poss_labels or c2 not in poss_labels:
        raise Exception("c1 or c2 whack! One is not in 'full_revert', 'partial_revert', 'non_revert'")

    X, y = get_prepped_csudf(path, to_numpy=False, multi_label=True, multi_file=multi_file)
    Xy = y.join(X)
    X_c1 = Xy.loc[Xy[c1] == True, :]
    X_c2 = Xy.loc[Xy[c2] == True, :]
    if len(X_c1.index) < len(X_c2.index):
        X_c2 = X_c2.sample(len(X_c1.index))
    else:
        X_c1 = X_c1.sample(len(X_c2.index))

    X = pd.concat([X_c1, X_c2])
    y = X[c1]
    X = X.drop(labels=['non_revert', 'partial_revert', 'full_revert'], axis=1)

    if to_numpy:
        y = y.to_numpy()
        X = X.to_numpy()
    return X, y


def find_first_el_index(df, first_el_var='version'):
    for i, col in enumerate(df.columns):
        if first_el_var + '_0' == col:
            return i
    return -1

def add_experience(df, variable, threshold):
    df['experience'] = df[variable] > threshold
    return df

def get_by_id(path, id, ovid_data=False, original_data=False):
    if ovid_data:
        X = pd.read_csv(path + 'ovid_data.csv').set_index('cs_id')
        y = pd.read_csv(path + 'labels.csv').set_index('cs_id')
        if original_data:
            X.drop(columns=['nprev_changesets', 'imagery_used'], inplace=True)
        else:
            X.drop(columns='acc_created', inplace=True)
    else:
        X, y = get_prepped_csudf(path)
    _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X.loc[id, :].to_numpy(), y.loc[id, 'label'], id in y_test.index






