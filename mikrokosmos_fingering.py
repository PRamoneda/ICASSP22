"""
    File name: mikrokosmos_fingering.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""

import glob
import json
import os
import sys

import pianoplayer.core


def load_json(name_file):
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


def load_xmls():
    grade, name, xmls = [], [], [],
    for path, g in load_json("mikrokosmos/mikrokosmos.json").items():
        grade.append(g)
        name.append(path)
        xmls.append(path)
    return zip(grade, name, xmls)


import multiprocessing as mp


def run_loop(args):
    r_h, l_h, xml = args

    pianoplayer.core.run_annotate(xml, outputfile=r_h, n_measures=800, depth=9,
                         right_only=True, quiet=False)
    pianoplayer.core.run_annotate(xml, outputfile=l_h, n_measures=800, depth=9,
                         left_only=True, quiet=False)
    print(f"done {xml}")


def do_fingering():
    num_workers = 2# mp.cpu_count()
    print("num_workers", num_workers)
    args = []
    for grade, path, xml in load_xmls():
        r_h_out = '/'.join(["Fingers/pianoplayer", os.path.basename(xml[:-4]) + '_rh.txt'])
        l_h_out = '/'.join(["Fingers/pianoplayer", os.path.basename(xml[:-4]) + '_lh.txt'])

        if not os.path.exists(r_h_out) or not os.path.exists(l_h_out):
            print(xml)
            args.append((r_h_out, l_h_out, xml))
        else:
            print("YA HA SIDO COMPUTADO")
    p = mp.Pool(processes=num_workers)
    p.map(run_loop, args)


if __name__ == "__main__":
    if not os.path.exists('Fingers/pianoplayer'):
        os.mkdir('Fingers/pianoplayer')
    do_fingering()
