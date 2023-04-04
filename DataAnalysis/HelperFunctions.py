import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance


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


def split_train_test_df(df, type, train_ids, test_ids, add_split_label=False, prep_data=False, to_numpy=False):
    if add_split_label:
        df['split'] = np.where(df['cs_id'].isin(train_ids['cs_id']), 'train', 'test')
    if type == "element":
        raise Exception("eldf not supported yet")
        ## TODO. antingen fixa så att de är sorterade i cs_ids, eller göra en vektor av matriser? Dock har matriserna olika storlek?
        #train_df = df.loc[train_ids.index, :].reindex(train_ids.index)
        #test_df = df.loc[test_ids.index, :].reindex(test_ids.index)
    else:
        train_df = df.loc[train_ids.index, :].reindex(train_ids.index)
        test_df = df.loc[test_ids.index, :].reindex(test_ids.index)

    if prep_data:
        train_df = prep_df(train_df, type, to_numpy=to_numpy)
        test_df = prep_df(test_df, type, to_numpy=to_numpy)
    return train_df, test_df

def split_train_test(labels, csdf=None, udf=None, eldf=None, test_size=0.3, add_split_label=False, prep_data=False, to_numpy=False):
    train_ids, test_ids = train_test_split(labels, test_size=test_size)
    train_csdf, train_udf, train_eldf, test_csdf, test_udf, test_eldf = None, None, None, None, None, None

    if csdf is not None:
        train_csdf, test_csdf = split_train_test_df(csdf, 'changeset', train_ids, test_ids, add_split_label=add_split_label, prep_data=prep_data, to_numpy=to_numpy)
    if udf is not None:
        train_udf, test_udf = split_train_test_df(udf, 'user', train_ids, test_ids, add_split_label=add_split_label, prep_data=prep_data, to_numpy=to_numpy)
    if eldf is not None:
        ## TODO. No support yet
        raise Exception("eldf Not supported yet")
        #train_eldf, test_eldf = split_train_test_df(eldf, 'element', train_ids, test_ids, add_split_label=add_split_label, prep_data=prep_data)
    if prep_data:
        train_ids = prep_df(train_ids, 'labels', to_numpy=to_numpy)
        test_ids = prep_df(test_ids, 'labels', to_numpy=to_numpy)

    return train_ids, train_csdf, train_udf, train_eldf, test_ids, test_csdf, test_udf, test_eldf


def prep_df(df, type, keep_csids=False, trim_only_csids=False, to_numpy=False):

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
    if trim_only_csids:
        del prepped['cs_id']
        return prepped.to_numpy() if to_numpy else prepped

    prepped = prepped.loc[:, ~prepped.columns.isin(to_trim)]
    for var in categoricals:
        if var in binary_cat:
            prepped.loc[:, var] = np.where(prepped.loc[:, var] == True, 1, 0)
        else:
            encoder = OneHotEncoder()
            encoded_df = pd.DataFrame(encoder.fit_transform(prepped[[var]]).toarray())
            encoded_df.index = prepped.index
            encoded_df.columns = encoder.get_feature_names_out()
            prepped = pd.concat([prepped, encoded_df], axis=1)
            del prepped[var]

    return prepped.to_numpy() if to_numpy else prepped


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

def random_forest(train_labels, test_labels, train_data, test_data, print_report=True, rfc=None):
    if rfc is None:
        rfc = RandomForestClassifier(n_estimators=100, max_depth=15, max_features=6, min_samples_leaf=2)
    rfc.fit(train_data, train_labels.ravel())
    pred_train = rfc.predict(train_data)
    pred_test = rfc.predict(test_data)
    if print_report:
        print('######################## REPORT FOR TRAINING DATA ###########################')
        print(classification_report(train_labels, pred_train, target_names=['Not Reverted', 'Reverted']))
        print('########################## REPORT FOR TEST DATA #############################')
        print(classification_report(test_labels, pred_test, target_names=['Not Reverted', 'Reverted']))
    return rfc

def get_feature_names(cs=True, u=True, el=False):
    names = []
    if cs:
        for feature in ['create', 'modify', 'delete', 'edits', 'nnodes', 'nways', 'nrelations',
                        'min_lon', 'max_lon', 'min_lat', 'max_lat', 'box_size', 'comment_len',
                        'imagery_used', 'editor_app_go map!!', 'editor_app_josm', 'editor_app_osm go!',
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

def rfc_importances(X_test, y_test, rfc, cs=True, u=True, el=False):
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

    forest_importances = pd.Series(importances)
    fig, ax = plt.subplots(2, 1, sharex='all')
    forest_importances.plot.bar(yerr=std, ax=ax[0])
    ax[0].set_title("Feature importances using MDI")
    ax[0].set_ylabel("Mean decrease in impurity")
    ax[0].set_title("Mean decrease in impurity")

    result = permutation_importance(rfc, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=get_feature_names(cs=cs, u=u, el=el))

    forest_importances.plot.bar(yerr=result.importances_std, ax=ax[1])
    ax[1].set_title("Feature importances using permutation on full model")
    ax[1].set_ylabel("Mean accuracy decrease")
    ax[1].set_title("Permutation importance")
    fig.tight_layout()
    fig.show()












