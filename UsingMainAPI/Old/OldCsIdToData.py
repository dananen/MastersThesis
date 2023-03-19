from time import strftime

import osmapi
import pprint
import requests
from bs4 import BeautifulSoup
import MainAPIHelperFunctions

def create_changeset_dict():
    return {'uid': 0,
            'id': 0,
            'create': 0,
            'modify': 0,
            'delete': 0,
            'edits': 0,
            'min_lon': 0,
            'max_lon': 0,
            'min_lat': 0,
            'max_lat': 0,
            'box_size': 0,
            'comment_len': 0,
            'created_at': 0,
            'imagery_used': False,
            'editor_app': 'other'}

def create_user_dict():
    return {'id': 0,
            'create': 0,
            'modify': 0,
            'delete': 0,
            'contributions': 0,
            'nprev_changesets': 0,
            'active_weeks': 0,
            'account_created': 0}
            #'nkeywords': 0,
            #'cs_above_500': False

def create_feature_dict():
    return {'changeset': 0,
            'type': '',
            'id': 0,
            'version_nbr': 0,
            'ntags': 0,
            'nprev_tags': 0,
            'nvalid_tags': 0,
            'nprev_valid_tags': 0,
          #  'nprev_authors': 0,
            'weeks_to_prev': 0,
            'name_changed': False,
            'operation': ''}


def cs_id_to_data(reverted_cs_id, api, VALID_TAGS, EDITOR_APPS,
                  query_user_data=True, max_cs_history=100, max_cs_objects=1000, max_nchanges=50000):
    reverted_cs = api.ChangesetGet(reverted_cs_id)
    reverted_cs_data = api.ChangesetDownload(reverted_cs_id)

    try:
        user_all_cs = api.ChangesetsGet(userid=reverted_cs["uid"], created_before=reverted_cs['created_at'])
    except:
        print("Skipped CS id: {0}. Can't download previous CS.".format(reverted_cs_id))
        return {}, {}, []
    current_cs_time = user_all_cs[next(iter(user_all_cs.keys()))]['created_at']

    total_nchanges = sum([int(cs['changes_count']) for cs in user_all_cs.values()])
    if total_nchanges > 50000:
        print("Skipped CS id: {0}. Previous CS exceeded {1} edits.".format(reverted_cs_id, max_nchanges))
        return {}, {}, []

    changeset_data = create_changeset_dict()
    user_data = create_user_dict()
    feature_data_list = []

    if 'comment' in reverted_cs['tag'].keys():
        changeset_data['comment_len'] = len(reverted_cs['tag']['comment'])
    if 'imagery_used' in reverted_cs['tag'].keys():
        changeset_data['imagery_used'] = True
    if 'created_by' in reverted_cs['tag'].keys():
        editor = reverted_cs['tag']['created_by'].lower()
        for editor_app in EDITOR_APPS:
            if editor_app in editor:
                changeset_data['editor_app'] = editor_app
                break

    try:
        changeset_data['id'] = reverted_cs_id
        changeset_data['uid'] = reverted_cs["uid"]
        changeset_data['min_lat'] = float(reverted_cs['min_lat'])
        changeset_data['min_lon'] = float(reverted_cs['min_lon'])
        changeset_data['max_lat'] = float(reverted_cs['max_lat'])
        changeset_data['max_lon'] = float(reverted_cs['max_lon'])
        changeset_data['box_size'] = (changeset_data['max_lon'] - changeset_data['min_lon']) * (
                    changeset_data['max_lat'] - changeset_data['min_lat'])
        changeset_data['created_at'] = MainAPIHelperFunctions.datetime_to_string(reverted_cs['created_at'])
    except Exception as e:
        print("Skipped CS {0}. Found exception {1}. Unable to find tag in changeset XML.".format(reverted_cs_id, e))
        return {}, {}, []

    try:

        for i, object_data_wrapper in enumerate(reverted_cs_data):

            if i+1 == max_cs_objects:
                break
            feature_data = create_feature_dict()
            feature_data['changeset'] = reverted_cs_id
            feature_data['type'] = object_data_wrapper['type']
            feature_data['operation'] = object_data_wrapper['action']
            object_data = object_data_wrapper['data']
            feature_data['id'] = object_data['id']
            changeset_data[object_data_wrapper['action']] += 1
            feature_data['version_nbr'] = object_data['version']
            tags = object_data['tag']
            feature_data['ntags'] = len(tags)
            for key, value in tags.items():
                if key in VALID_TAGS.keys() and value in VALID_TAGS[key]:
                    feature_data['nvalid_tags'] += 1

            # for looking only at the previous version
            if object_data['version'] != 1:
                if object_data_wrapper['type'] == 'node':
                    prev_version = api.NodeGet(object_data['id'], object_data['version']-1)
                elif object_data_wrapper['type'] == 'way':
                    prev_version = api.WayGet(object_data['id'], object_data['version']-1)
                elif object_data_wrapper['type'] == 'relation':
                    prev_version = api.RelationGet(object_data['id'], object_data['version']-1)

                feature_data['weeks_to_prev'] = (object_data['timestamp'] - prev_version['timestamp']).days/7

                prev_tags = prev_version['tag']
                feature_data['nprev_tags'] = len(prev_tags)
                for key, value in prev_tags.items():
                    if key in VALID_TAGS.keys() and value in VALID_TAGS[key]:
                        feature_data['nprev_valid_tags'] += 1

                reverted_cs_name, prev_version_name = '', ''
                if 'name' in object_data['tag'].keys():
                    reverted_cs_name = object_data['tag']['name']
                if 'name' in prev_version['tag'].keys():
                    prev_version_name = prev_version['tag']['name']
                feature_data['name_changed'] = reverted_cs_name != prev_version_name


            # for looking at the entire object history
            # if object_data_wrapper['type'] == 'node':
            #     object_history = api.NodeHistory(object_data['id'])
            # elif object_data_wrapper['type'] == 'way':
            #     object_history = api.WayHistory(object_data['id'])
            # elif object_data_wrapper['type'] == 'relation':
            #     object_history = api.RelationHistory(object_data['id'])
            #
            # prev_authors = set()
            # previous, before_prev = False, False
            # reverted_cs_time, reverted_cs_name = 0, ''
            # nvalid_tags, nprev_valid_tags = 0, 0
            # for object_version in reversed(object_history.values()):
            #     if before_prev:
            #         prev_authors.add(object_version['uid'])
            #     elif previous:
            #         prev_authors.add(object_version['uid'])
            #         feature_data['weeks_to_prev'] = (reverted_cs_time - object_version['timestamp']).days/7
            #         prev_tags = object_version['tag']
            #         feature_data['nprev_tags'] = len(prev_tags)
            #         for key, value in prev_tags.items():
            #             if key in VALID_TAGS.keys() and value in VALID_TAGS[key]:
            #                 nprev_valid_tags += 1
            #         feature_data['nprev_valid_tags'] = nprev_valid_tags
            #         if 'name' in object_version['tag'].keys():
            #             feature_data['name_changed'] = reverted_cs_name != object_version['tag']['name']
            #         else:
            #             feature_data['name_changed'] = reverted_cs_name != ''
            #         previous = False
            #         before_prev = True
            #     elif object_version['changeset'] == reverted_cs_id:
            #         reverted_cs_time = object_version['timestamp']
            #         tags = object_version['tag']
            #         feature_data['ntags'] = len(tags)
            #         for key, value in tags.items():
            #             if key in VALID_TAGS.keys() and value in VALID_TAGS[key]:
            #                 nvalid_tags += 1
            #         feature_data['nvalid_tags'] = nvalid_tags
            #         if 'name' in object_version['tag'].keys():
            #             reverted_cs_name = object_version['tag']['name']
            #         previous = True
            #
            # feature_data['nprev_authors'] = len(prev_authors)
            # if reverted_cs['uid'] in prev_authors:
            #     feature_data['nprev_authors'] -= 1

            feature_data_list.append(feature_data)

        changeset_data['edits'] = changeset_data['create'] + changeset_data['modify'] + changeset_data['delete']
    except Exception as e:
        print("Skipped id {0}. Found exception {1} while collecting feature data.".format(reverted_cs_id, e))
        return {}, {}, []

    # keywords = {'building', 'source', 'highway', 'name', 'natural', 'surface', 'landuse', 'power', 'waterway', 'amenity', 'service', 'oneway'}
    # nkeywords = 0
    if query_user_data:
        try:
            user_data['id'] = reverted_cs['uid']
            soup = BeautifulSoup(
                requests.get("https://api.openstreetmap.org/api/0.6/user/" + str(reverted_cs['uid'])).content,
                'lxml-xml')
            soup = soup.find('user')
            if soup != None and 'account_created' in soup.attrs.keys():
                user_data['account_created'] = soup.attrs['account_created']
            #    changeset_data['weeks_account_active'] = (changeset_data['created_at'] - user_data['account_created']).days/7
            #  if soup != None and 'count' in soup.attrs.keys():
            #      user_data['nchangesets'] = soup.attrs['count']

            nprev_changesets, nactive_weeks, reached_last = 0, 0, False
            while nprev_changesets < max_cs_history and not reached_last:
                for id, cs in user_all_cs.items():
                    if nprev_changesets == max_cs_history:
                        break
                    nprev_changesets += 1
                    cs_data = api.ChangesetDownload(id)
                    for object_data_wrapper in cs_data:
                        user_data[object_data_wrapper['action']] += 1

                    if (current_cs_time - cs['created_at']).days >= 7:
                        current_cs_time = cs['created_at']
                        nactive_weeks += 1

                   # print(nprev_changesets)

                    # if 'comment' in cs['tag'].keys(): keywords blir whack för långt ifrån alla skriver på engelska?
                    #    for word in cs['tag']['comment']:
                    #        if word in keywords:
                    #            nkeywords += 1

                if nprev_changesets < max_cs_history:
                    if len(user_all_cs) == 100:
                        user_all_cs = api.ChangesetsGet(userid=reverted_cs["uid"],
                                                        created_before=user_all_cs[next(reversed(user_all_cs.keys()))]['created_at'])
                    else:
                        reached_last = True

            user_data['cs_above_' + str(max_cs_history)] = not reached_last
            user_data['nprev_changesets'] = nprev_changesets
            user_data['active_weeks'] = nactive_weeks
            user_data['contributions'] = user_data['create'] + user_data['modify'] + user_data['delete']
        except Exception as e:
            print("Skipped id {0}. Found exception {1} while collecting user data.".format(reverted_cs_id, e))
            return {}, {}, []

    return changeset_data, user_data, feature_data_list

