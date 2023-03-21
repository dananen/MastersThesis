import csv
from copy import deepcopy
import datetime
import string


def get_valid_tags():
    rd = csv.reader(open('/Users/dansvenonius/Desktop/Ovid/training_sample/map_feature_list.tsv'), delimiter="\t")
    VALID_TAGS = {}
    for row in rd:
        if row[0] == 'key':
            continue
        if row[0] not in VALID_TAGS.keys():
            VALID_TAGS[row[0]] = [row[1]]
        else:
            VALID_TAGS[row[0]].append(row[1])
    for key, value in deepcopy(VALID_TAGS).items():
        if len(value) < 3:
            del VALID_TAGS[key]
    return VALID_TAGS


def get_editor_apps():
    return {'potlatch', 'josm', 'vespucci', 'osm go!', 'go map!!', 'streetcomplete'}


def get_top_12_keywords():
    return {'building', 'source', 'highway', 'name', 'natural', 'surface', 'landuse', 'power', 'waterway', 'amenity', 'service', 'oneway'}


def containsID(comment):
    # Returnar alla idn i en kommentar i form av en lista, ifall ordet innan var 'changeset' || 'chgset' || 'cs'
    comment = comment.lower().translate(str.maketrans('', '', string.punctuation)).split(' ')
    return [int(comment[i]) for i in range(len(comment)) if
            comment[i].isnumeric() and (len(comment[i]) == 8 or len(comment[i]) == 9)
            and i > 0 and comment[i-1] in {'changeset', 'chgset', 'cs'}]


def add_dict(big, small):
    if not small:
        return big
    if not big:
        return {key: [value] for key, value in small.items()}
    for key in big.keys():
        big[key].append(small[key])
    return big


def string_to_datetime(time_string, conversion='%Y-%m-%d %H:%M:%S%z', iso=False):
    if iso:
        return datetime.datetime.fromisoformat(time_string)
    return datetime.datetime.strptime(time_string, conversion)


def datetime_to_string(datetime_object, conversion='%Y-%m-%d %H:%M:%S%z'):
    return datetime_object.strftime(conversion)


def osmchange_to_changeset_data(cs_id, cs_created_at, cs, osmchange_data, VALID_TAGS):
    changeset_data = {'uid': 0,
                        'cs_id': 0,
                        'create': 0,
                        'modify': 0,
                        'delete': 0,
                        'edits': 0,
                        'nnodes': 0,
                        'nways': 0,
                        'nrelations': 0,
                        'min_lon': 0,
                        'max_lon': 0,
                        'min_lat': 0,
                        'max_lat': 0,
                        'box_size': 0,
                        'comment_len': 0,
                        'created_at': 0,
                        'imagery_used': False,
                        'editor_app': 'other'}

    try:
        changeset_data['uid'] = cs['uid']
        changeset_data['cs_id'] = cs_id
        changeset_data['min_lat'] = float(cs['min_lat'])
        changeset_data['min_lon'] = float(cs['min_lon'])
        changeset_data['max_lat'] = float(cs['max_lat'])
        changeset_data['max_lon'] = float(cs['max_lon'])
        changeset_data['box_size'] = (changeset_data['max_lon'] - changeset_data['min_lon']) * (
                    changeset_data['max_lat'] - changeset_data['min_lat'])
        #changeset_data['created_at'] = datetime_to_string(cs['created_at'])
        changeset_data['created_at'] = cs_created_at
    except Exception as e:
        print("Skipped CS {0}. Found exception {1}. Unable to find tag in changeset XML.".format(cs_id, e))
        return {}, []

    if 'comment' in cs['tag'].keys():
        changeset_data['comment_len'] = len(cs['tag']['comment'])
    if 'imagery_used' in cs['tag'].keys():
        changeset_data['imagery_used'] = True
    if 'created_by' in cs['tag'].keys():
        editor = cs['tag']['created_by'].lower()
        for editor_app in get_editor_apps():
            if editor_app in editor:
                changeset_data['editor_app'] = editor_app
                break

    dict_list = []
    for cs_object in osmchange_data:
        changeset_data[cs_object['action']] += 1

        if 'type' in cs_object.keys() and 'id' in cs_object['data'].keys() and 'version' in cs_object['data'].keys():
            changeset_data['n' + cs_object['type'] + 's'] = cs_object['type']
            cs_object_data = cs_object['data']
            element_dict = {'cs_id': cs_id,
                            'uid': changeset_data['uid'],
                            'type': cs_object['type'],
                            'element_id': cs_object_data['id'],
                            'version': cs_object_data['version'],
                            'operation': cs_object['action']}
            if element_dict['version'] == 1:
                tags = cs_object_data['tag']
                element_dict['ntags'] = len(tags)
                element_dict['nvalid_tags'] = len([k for k, v in tags.items() if k in VALID_TAGS.keys() and v in VALID_TAGS[k]])
                element_dict['nprev_tags'] = 0
                element_dict['nprev_valid_tags'] = 0
                element_dict['weeks_to_prev'] = 0
                element_dict['name_changed'] = False
            dict_list.append(element_dict)

    changeset_data['edits'] = changeset_data['create'] + changeset_data['modify'] + changeset_data['delete']
    return changeset_data, dict_list





