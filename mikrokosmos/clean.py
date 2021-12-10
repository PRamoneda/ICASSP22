import csv
import os
import re


def clean():
    input_file = csv.DictReader(open("mikrokosmos_metadata.csv"))
    for piece in input_file:
        if piece["4 hands"] == 'FALSE':
            print(piece['piece number'])
            f = open(f"musicxml/{piece['piece number']}.xml", "r")
            data = f.read()
            # data = re.sub(' *<software>Neuratron PhotoScore</software>\n', '', data)
            # data = re.sub('<source>.*</source>',
            #        '<source>Mikrokosmos expired copy-right edition. Transcribed by Pedro Ramoneda to MusicXML</source>\n',
            #        data)
            # work = piece['work'].replace('\\u00e9', 'é').replace('\\u00e0', '')
            # data = re.sub('<work-title></work-title>', f"<work-title>{work}, Mikrokosmos, Sz. 107</work-title>", data)
            # data = re.sub('<creator type="composer"></creator>', f"<creator type=\"composer\">Béla Bártok</creator>", data)
            # data = re.sub('<creator type="composer">Béla Bártok</creator>', f'<creator type="composer">Béla Bartók</creator>',
            #               data)
            # data = re.sub('<part-name>.*</part-name>', '<part-name></part-name>', data)
            data = re.sub('<instrument-name>.*</instrument-name>', '<instrument-name></instrument-name>', data)
            # data = re.sub('<part-abbreviation>.*</part-abbreviation>', '<part-abbreviation></part-abbreviation>', data)
            f = open(f"musicxml/{piece['piece number']}.xml", "w")
            f.write(data)




clean()