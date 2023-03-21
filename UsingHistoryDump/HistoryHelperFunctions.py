import pandas as pd
import datetime
from UsingMainAPI import MainAPIHelperFunctions as mainhf


def init_big_element_dict(path, type):
    df = pd.read_csv(path + type + '_data.csv')

    # Vi har en stor map med alla elements id. de mappar till changeset_ids som i sin tur mappar till ett changeset-objekt
    # Detta för att samma objekt kan ha editerats i flera olika changesets
    big_dict = {}
    for index, row in df.iterrows():
        if index%10000 == 0:
            print(index)
        if row['element_id'] in big_dict.keys():
            if row['cs_id'] in big_dict[row['element_id']].keys():
                big_dict[row['element_id']][row['cs_id']]['version'].append(row['version'])
            else:
                big_dict[row['element_id']][row['cs_id']] = init_nested_element_dict(type, row)
        else:
            big_dict[row['element_id']] = {row['cs_id']: init_nested_element_dict(type, row)}
    return big_dict


def init_nested_element_dict(element_type, df_row):
    return {'cs_id': df_row['cs_id'],
            'type': element_type,
            'id': df_row['element_id'],
            'version': [df_row['version']],
            'operation': df_row['operation'],
            'ntags': 0,
            'nprev_tags': 0,
            'nvalid_tags': 0,
            'nprev_valid_tags': 0,
            'weeks_to_prev': 0,
            'name_changed': False,
            'nprev_auths': 0}


def init_user_dict(file):
    df = pd.read_csv(file)
    big_dict = {}
    for index, row in df.iterrows():
        if index % 10000 == 0:
            print(index)
        if row['uid'] in big_dict.keys():
            big_dict[row['uid']][row['cs_id']] = init_nested_user_dict(row)
        else:
            big_dict[row['uid']] = {row['cs_id']: init_nested_user_dict(row)}
    return big_dict


def init_nested_user_dict(row):
    d = {'uid': row['uid'],
            'cs_id': row['cs_id'],
            'cs_created_at': mainhf.string_to_datetime(row['cs_created_at']),
            'create': 0,
            'modify': 0,
            'delete': 0,
            'contributions': 0,
            'nprev_changesets': 0,
            'active_weeks': 0,
            'acc_created': '',
            'create_nodes': 0,
            'create_ways': 0,
            'create_relations': 0,
            'create_building': 0,
            'create_source': 0,
            'create_highway': 0,
            'create_name': 0,
            'create_natural': 0,
            'create_surface': 0,
            'create_landuse': 0,
            'create_power': 0,
            'create_waterway': 0,
            'create_amenity': 0,
            'create_service': 0,
            'create_oneway': 0}
    if 'create' in row:
        d['create'] = row['create']
    if 'modify' in row:
        d['modify'] = row['modify']
    if 'delete' in row:
        d['delete'] = row['delete']
    if 'contributions' in row:
        d['contributions'] = row['contributions']
    if 'create_nodes' in row:
        d['create_nodes'] = row['create_nodes']
    if 'create_ways' in row:
        d['create_ways'] = row['create_ways']
    if 'create_relations' in row:
        d['create_relations'] = row['create_relations']
    if 'nprev_changesets' in row:
        d['nprev_changesets'] = row['nprev_changesets']
    if 'active_weeks' in row:
        d['active_weeks'] = row['active_weeks']
    if 'acc_created' in row:
        d['acc_created'] = row['acc_created']
    return d


def process_element_history(element_history, element_dicts):
    for cs_id, element_dict in element_dicts.items():
        reverted_indices = []
        for i in range(len(element_history)):
            if 'version' in element_history[i].keys() and element_history[i]['version'] in element_dict['version']:
                reverted_indices.append(i)
        min_idx = min(reverted_indices)
        if len(element_dicts) > 1:
            pass
        if min_idx == 0 or element_history[min_idx]['version'] != element_history[min_idx-1]['version']+1:
            element_dicts[cs_id] = {'cs_id': cs_id}
            break

        prev_iteration = element_history[min_idx-1]
        last_changeset_iteration = element_history[max(reverted_indices)]

        prev_auths = set()
        for i in range(min_idx+1):
            prev_auths.add(element_history[i]['author'])

        element_dict['version'] = max(element_dict['version'])
        element_dict['ntags'] = last_changeset_iteration['ntags']
        element_dict['nprev_tags'] = prev_iteration['ntags']
        element_dict['nvalid_tags'] = last_changeset_iteration['nvalid_tags']
        element_dict['nprev_valid_tags'] = prev_iteration['nvalid_tags']
        element_dict['weeks_to_prev'] = (last_changeset_iteration['timestamp'] - prev_iteration['timestamp']).total_seconds()/(60*60*24*7)
        element_dict['name_changed'] = last_changeset_iteration['names'] != prev_iteration['names']
        #element_dict['operation'] = determine_operation(prev_iteration['visible'], changeset_iteration['visible'])
        element_dict['nprev_auths'] = len(prev_auths)-1 #excluding reverted author
    return element_dicts


def process_element_iteration(element, reverted_versions, VALID_TAGS):
    element_iteration = {'version': element.version, 'author': element.uid}
    if element.version + 1 in reverted_versions or element.version in reverted_versions:
        element_iteration['visible'] = element.visible
        element_iteration['timestamp'] = element.timestamp
        tags = element.tags
        element_iteration['ntags'] = len(tags)
        valid_tags, names = 0, {}
        for tag in tags:
            key, value = tag.k, tag.v
            if 'name' in key:
                names[key] = value
            if key in VALID_TAGS.keys() and value in VALID_TAGS[key]:
                valid_tags += 1
        element_iteration['names'] = names
        element_iteration['nvalid_tags'] = valid_tags
    return element_iteration


def determine_operation(prev_element_visible, curr_element_visible, version=0):
    if version == 1:
        return 'create'
    return 'delete' if not curr_element_visible and prev_element_visible else 'modify'


def nbr_keywords_added(prev_tags, curr_tags):
    if len(curr_tags) == 0:
        return {}
    nbr_tags = {}
    for key, val in curr_tags.items():
        if (key, val) not in prev_tags.items():
            if key in nbr_tags.keys():
                nbr_tags[key] += 1
            else:
                nbr_tags[key] = 1
    return nbr_tags

