import csv
import json
import os


def slice():
    input_file = csv.DictReader(open("mikrokosmos_metadata.csv"))
    for piece in input_file:
        if piece["4 hands"] == 'FALSE' and piece["piece number"] in ['151']:
            os.system(f"gs -dBATCH -dNOPAUSE -sOutputFile=pdfs/{piece['piece number']}.pdf -dFirstPage={piece['start_pdf']} -dLastPage={piece['end_pdf']} -sDEVICE=pdfwrite mikrokosmos_full.pdf")

# slice()


def save_json(dictionary, name_file):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4, ensure_ascii=False)


TO_LABEL = {
    "Mikrokosmos, Volumes I-II": 0,
    "Mikrokosmos, Volumes III-IV": 1,
    "Mikrokosmos, Volumes V-VI": 2
}

def mikro_json():
    input_file = csv.DictReader(open("mikrokosmos_metadata.csv"))
    data = {f"mikorkosmos/musicxml/{piece['piece number']}.xml": TO_LABEL[piece['book']]
            for piece in input_file if piece["4 hands"] == 'FALSE'}
    save_json(data, 'mikrokosmos.json')


# mikro_json()

def henle_mikro_json():
    input_file = csv.DictReader(open("mikrokosmos_metadata.csv"))
    data = {int(piece['piece number']): int(piece['henle_difficulty'][6])
            for piece in input_file if piece["4 hands"] == 'FALSE'}
    save_json(data, 'henle_mikrokosmos.json')

henle_mikro_json()