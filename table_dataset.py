"""
    File name: approach_deepgru.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""

import os.path
from statistics import mean, stdev

import music21
from music21 import converter, stream

from ScoreFeat.scorefeat import ScoreFeat
from utils import load_json


def num_notes(path):
    idx = os.path.basename(path)[:-4]
    file = open(f"Fingers/mikrokosmos2/{idx}_lh.txt", "r")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    line_count_lh = len(nonempty_lines)
    file.close()
    file = open(f"Fingers/mikrokosmos2/{idx}_rh.txt", "r")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    line_count_rh = len(nonempty_lines)
    file.close()
    return line_count_lh + line_count_rh

data = load_json("mikrokosmos/mikrokosmos.json")
stats = {
    0: {"scores": []},
    1: {"scores": []},
    2: {"scores": []}
}
for k, v in data.items():
    stats[v]["scores"] = stats[v]["scores"] + [k]

for lvl in [0, 1, 2]:
    stats[lvl]['n_scores'] = len(stats[lvl]['scores'])
    print(f"lvl: {lvl}; number of scores: {len(stats[lvl]['scores'])}")


for lvl in [0, 1, 2]:
    ps_rate, notes, pf_rate, pe_rate, measures, tempo = [], [], [], [], [], []
    for path in stats[lvl]['scores']:
        sc = converter.parse(path)
        notes.append(num_notes(path))
        measures.append(len(sc.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure)))
        tempo.append(list(sc.flat.getElementsByClass(music21.tempo.MetronomeMark))[0].number)
        sf = ScoreFeat(path)
        # ps_rate.append(sf.playing_speed_per_s())
        # pf_rate.append(sf.polyphonic_rate())
        # pe_rate.append(sf.pitch_entropy())
    print(f"lvl: {lvl}; notes: {mean(notes)} {stdev(notes)}")
    print(f"lvl: {lvl}; measure: {mean(measures)} {stdev(measures)}")
    print(f"lvl: {lvl}; tempo: {mean(tempo)} {stdev(tempo)}")
    # print(f"lvl: {lvl}; pr: {mean(pf_rate)} {stdev(pf_rate)}")
    # print(f"lvl: {lvl}; pe: {mean(pe_rate)} {stdev(pe_rate)}")
    # print(f"lvl: {lvl}; ps: {mean(pe_rate)} {stdev(pe_rate)}")



