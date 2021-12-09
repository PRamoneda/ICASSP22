import json



import json
import pickle


def save_json(dictionary, name_file):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def load_xmls():
    grade, name, xmls = [], [], [],
    for path, g in load_json("mikrokosmos/mikrokosmos.json").items():
        grade.append(g)
        name.append(path)
        xmls.append(path)
    return zip(grade, name, xmls)


def strm2map(strm):
    converted = []
    om = []
    for o in strm.flat.secondsMap:
        if o['element'].isClassOrSubclass(('Note',)):
            om.append(o)
        elif o['element'].isClassOrSubclass(('Chord',)):
            om_chord = [{'element': oc,
                         'offsetSeconds': o['offsetSeconds'],
                         'endTimeSeconds': o['endTimeSeconds'],
                         'chord': o['element']} for oc in sorted(o['element'].notes, key=lambda a: a.pitch)]
            om.extend(om_chord)
    om_filtered = []
    for o in om:
        offset = o['offsetSeconds']
        duration = o['endTimeSeconds']
        pitch = o['element'].pitch
        simultaneous_notes = [o2 for o2 in om if o2['offsetSeconds'] == offset and o2['element'].pitch.midi == pitch.midi]
        max_duration = max([float(x['endTimeSeconds']) for x in simultaneous_notes])
        if len(simultaneous_notes) > 1 and duration < max_duration and str(offset) + ':' + str(pitch) not in converted:
            continue
        else:
            converted.append(str(offset) + ':' + str(pitch))

        if not (o['element'].tie and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop')) and \
                not ((hasattr(o['element'], 'tie') and o['element'].tie
                      and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop'))) and \
                not (o['element'].duration.quarterLength == 0):
            om_filtered.append(o)

    return sorted(om_filtered, key=lambda a: (a['offsetSeconds'], a['element'].pitch))