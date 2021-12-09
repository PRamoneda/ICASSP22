"""
    File name: approach_deepgru.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""


import csv
import os
import sys

import numpy as np

from utils import load_xmls, load_json, save_json


def get_path(alias):
    if alias == "mikro1":
        path = "mikrokosmos1"
    if alias == "mikro2":
        path = "pianoplayer"
    if alias == "nak":
        path = "nakamura"
    return path


def rep_raw(alias):
    path_alias = get_path(alias)
    rep = {}
    for grade, path, xml in load_xmls():
        rep[path] = {
            'grade': grade,
            'right_velocity': [],
            'left_velocity': [],
            'right_fingers': [],
            'left_fingers': []
        }
        r_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
        l_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])
        for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
            with open(path_txt) as tsv_file:
                read_tsv = csv.reader(tsv_file, delimiter="\t")
                for l in read_tsv:
                    rep[path][hand + 'velocity'] = rep[path][hand + 'velocity'] + [float(l[8])]
                    rep[path][hand + 'fingers'] = rep[path][hand + 'fingers'] + [abs(int(l[7]))]
    save_json(rep, os.path.join('representations', path_alias, 'rep_raw.json'))


def merge_chord_onsets(time_series):
    new_time_series = [list(a) for a in time_series]
    for ii in range(len(time_series)):
        if ii + 1 < len(time_series) and time_series[ii][0] + 0.05 == time_series[ii + 1][0]:
            if ii + 2 < len(time_series) and time_series[ii][0] + 0.1 == time_series[ii + 2][0]:
                if ii + 3 < len(time_series) and time_series[ii][0] + 0.15 == time_series[ii + 3][0]:
                    if ii + 4 < len(time_series) and time_series[ii][0] + 0.2 == time_series[ii + 4][0]:
                        new_time_series[ii][0] = time_series[ii + 4][0]
                    else:
                        new_time_series[ii][0] = time_series[ii + 3][0]
                else:
                    new_time_series[ii][0] = time_series[ii + 2][0]
            else:
                new_time_series[ii][0] = time_series[ii + 1][0]
        else:
            new_time_series[ii][0] = time_series[ii][0]
    return [tuple(a) for a in new_time_series]


def finger2index(f):
    if f > 0:
        index = int(f) + 4
    elif f < 0:
        index = int(f) - 5
    else:  # == 0
        index = -1000
    return index


def velocity_piece(path, alias, xml):
    path_alias = get_path(alias)
    print(path)
    r_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
    l_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])

    intermediate_rep = []
    for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
        time_series = []
        with open(path_txt) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for l in read_tsv:
                if int(l[7]) != 0:
                    time_series.append((round(float(l[1]), 2), int(l[7]), abs(float(l[8]))))
        time_series = time_series[:-9]
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = intermediate_rep[idx][2]
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = intermediate_rep[j][2]
            j += 1
        idx = j
        # print(t)
        matrix.append(t)
    return matrix, onsets


def rep_velocity(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = velocity_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_velocity.json'))


def prob_piece(path, alias, xml):
    path_alias = get_path(alias)

    print(path)
    PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '.txt'])

    time_series = []
    with open(PIG_cost) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for l in list(read_tsv)[1:]:
            if int(l[7]) != 0:
                time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8])))))
    time_series = time_series[:-3]

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

    onsets = []
    idx = 0
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = intermediate_rep[idx][2]
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = intermediate_rep[j][2]
            j += 1
        idx = j
        # print(t)
        matrix.append(t)

    return matrix, onsets


def rep_prob(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = prob_piece(path, alias, xml)

        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }
    save_json(rep, os.path.join('representations', get_path(alias), 'rep_nakamura.json'))


def rep_d_nakamura(alias):
    path_alias = get_path(alias)
    rep = {}
    for grade, path, xml in load_xmls():
        print(path, grade)
        PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '.txt'])

        time_series = []
        with open(PIG_cost) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for l in list(read_tsv)[1:]:
                if int(l[7]) != 0:
                    time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8]))), round(float(l[2]), 2)))
        time_series = time_series[:-3]


        # order by onset and create matrix
        matrix = []
        intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

        idx = 0
        while idx < len(intermediate_rep):
            t = [0] * 10
            index = finger2index(intermediate_rep[idx][1])

            t[index] = intermediate_rep[idx][2] / (intermediate_rep[idx][3] - intermediate_rep[idx][0])
            j = idx + 1
            while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
                index = finger2index(intermediate_rep[j][1])
                t[index] = intermediate_rep[j][2] / (intermediate_rep[j][3] - (intermediate_rep[j][0]))
                j += 1
            idx = j
            # print(t)
            matrix.append(t)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', path_alias, 'rep_d_nakamura.json'))


def finger_piece(path, alias, xml):
    path_alias = get_path(alias)
    print(path)
    r_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
    l_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])

    intermediate_rep = []
    for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
        time_series = []
        with open(path_txt) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for l in read_tsv:
                if int(l[7]) != 0:
                    time_series.append((round(float(l[1]), 2), int(l[7]), abs(float(l[8]))))
        time_series = time_series[:-1]
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = 1.0
            j += 1
        idx = j
        # print(t)
        matrix.append(t)

    return matrix, onsets

def rep_finger(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = finger_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_finger.json'))


def finger_nakamura_piece(path, alias, xml):
    path_alias = get_path(alias)
    print(path)
    PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '.txt'])

    time_series = []
    with open(PIG_cost) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for l in list(read_tsv)[1:]:
            if int(l[7]) != 0:
                time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8])))))
    time_series = time_series[:-3]

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = 1.0
            j += 1
        idx = j
        # print(t)
        matrix.append(t)
    return matrix, onsets


def rep_finger_nakamura(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = finger_nakamura_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_finger_nakamura.json'))


def notes_piece(path, alias, xml):
    path_alias = get_path(alias)
    print(path)
    r_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
    l_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])

    intermediate_rep = []
    for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
        time_series = []
        with open(path_txt) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for l in read_tsv:
                if int(l[7]) != 0:
                    # (onset, note)
                    time_series.append((round(float(l[1]), 2), int(l[3]) - 21))
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0.0] * 88
        index = intermediate_rep[idx][1]
        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = intermediate_rep[j][1]
            t[index] = 1.0
            j += 1
        idx = j
        # print(t)
        matrix.append(t)
    return matrix, onsets


def rep_notes(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = notes_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_note.json'))


def visualize_note_representation(alias, score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_note.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_finger_representation(alias, score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_finger.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_finger_representation_nakamura(alias="nak", score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_finger_nakamura.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_velocity_representation(alias, score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_velocity.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_prob_representation(alias="nak", score='mikrokosmos/musicxml/5.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_nakamura.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print('%02.1f' % c, end="|")
        print()


def visualize_d_nakamura(alias="nak", score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_d_nakamura.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def get_distance_type(last_semitone, current_semitone):
    last_black = (last_semitone % 12) in [1, 3, 6, 8, 10]
    current_black = (current_semitone % 12) in [1, 3, 6, 8, 10]

    if not last_black and not current_black:
        distance_type = 1
    elif last_black and not current_black:
        distance_type = 2
    elif not last_black and current_black:
        distance_type = 3
    else:  # bb
        distance_type = 4

    return distance_type


def rep_distances(alias):
    path_alias = get_path(alias)

    rep = {}
    for grade, path, r_h, l_h in load_xmls():
        print(path, grade)
        r_h_cost = '/'.join(["Fingers", path_alias, r_h[:-11] + '_rh.txt'])
        l_h_cost = '/'.join(["Fingers", path_alias, l_h[:-11] + '_lh.txt'])

        intermediate_rep = []
        for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
            time_series = []
            with open(path_txt) as tsv_file:
                read_tsv = csv.reader(tsv_file, delimiter="\t")
                for l in read_tsv:
                    if int(l[7]) != 0:
                        time_series.append((round(float(l[1]), 2), int(l[7]), abs(float(l[8])), abs(float(l[3]))))
            if alias == 'version_1.0':
                time_series = merge_chord_onsets(time_series[:-10])
            else:
                time_series = time_series[:-10]
            intermediate_rep.extend(time_series)

        # order by onset and create matrix
        matrix = []
        intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]
        # initial semitone: at the beginning the distance is 0
        last_semitone_rh = next(x[3] for x in intermediate_rep if x[1] > 0)
        last_semitone_lh = next(x[3] for x in intermediate_rep if x[1] < 0)
        idx = 0
        while idx < len(intermediate_rep):
            d, dt, t = [0] * 10, [0] * 10, [0] * 10
            index = finger2index(intermediate_rep[idx][1])
            is_r_h = index >= 5
            last_semitone = last_semitone_rh if is_r_h else last_semitone_lh
            t[index] = intermediate_rep[idx][2]
            d[index] = last_semitone - intermediate_rep[idx][3]
            dt[index] = get_distance_type(last_semitone, intermediate_rep[idx][3])
            if is_r_h:
                last_semitone_rh = last_semitone
            else:
                last_semitone_lh = last_semitone

            j = idx + 1
            while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
                index = finger2index(intermediate_rep[j][1])
                is_r_h = index >= 5
                last_semitone = last_semitone_rh if is_r_h else last_semitone_lh
                t[index] = intermediate_rep[j][2]
                d[index] = last_semitone - intermediate_rep[idx][3]
                dt[index] = get_distance_type(last_semitone, intermediate_rep[idx][3])
                if is_r_h:
                    last_semitone_rh = last_semitone
                else:
                    last_semitone_lh = last_semitone
                j += 1
            idx = j
            # print(t)
            # matrix.append([t, d , dt])
            matrix.append(t + d + dt)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }
    save_json(rep, os.path.join('representations', path_alias, 'rep_distance.json'))


def rep_fing_vel_time(alias):
    get_path(alias)


def rep_distances_time(alias):
    get_path(alias)


def rep_merged_time(alias):
    get_path(alias)


def load_rep(klass):
    if klass == "rep_velocity":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_velocity.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_finger":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_finger.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_finger_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_finger_nakamura.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_prob":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_nakamura.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_d_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_d_nakamura.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_note":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_note.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_distance":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_distance.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    return ans


def load_rep_info(klass):
    if klass == "rep_velocity":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_velocity.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_finger":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_finger.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_finger_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_finger_nakamura.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_prob":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_nakamura.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_d_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_d_nakamura.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_note":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_note.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_distance":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_distance.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    return ans




if __name__ == '__main__':
    # rep_raw("version_1.0")
    rep_velocity("mikro2")
    # rep_distances("version_1.0")
    # load_rep("version_1.0", rep_velocity)
    # load_rep("version_1.0", rep_velocity)
    # visualize_note_representation("mikro1")
    # visualize_note_representation("mikro1")
    # rep_finger_nakamura("nak")
    # rep_prob("nak")
    rep_notes("mikro2")
    rep_finger("mikro2")
    # visualize_note_representation("mikro2")
    # rep_d_nakamura("nak")
    # visualize_prob_representation("nak")
    # visualize_finger_representation_nakamura()
