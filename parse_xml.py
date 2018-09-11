import xml.etree.ElementTree as ET
import pathlib
import re
import os

DATA_DIR = pathlib.Path('ASK')

INPUT_DIR = DATA_DIR / 'xml'
OUTPUT_DIR = DATA_DIR / 'txt'

for filename in os.listdir(str(INPUT_DIR)):
    input_file = str(INPUT_DIR / filename)
    output_file = str(OUTPUT_DIR / filename)[:-3] + 'txt'

    tree = ET.parse(input_file)
    root = tree.getroot()

    metadata = root.find('./teiHeader/profileDesc/particDesc/person')

    language = metadata.find("./p[@n='language']").text
    cefr_score_node = metadata.find("./p[@n='CEFRscore']")
    cefr_score = 'N/A' if cefr_score_node is None else cefr_score_node.text

    text = root.find('text')

    with open(output_file, 'w') as outfile:
        outfile.write('# language: %s\n' % language)
        outfile.write('# CEFRscore: %s\n' % cefr_score)

        for paragraph in text.iter('p'):
            for sentence in paragraph.iter('s'):
                words = [w.text for w in sentence.iter('word')]
                joined = ' '.join(words)
                joined = re.sub(r'\s([.,?!:;])', r'\1', joined)
                outfile.write(joined + '\n')
            outfile.write('\n')
    print('Wrote ' + output_file)
