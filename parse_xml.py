import argparse
import pathlib
import re
import xml.etree.ElementTree as ET

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', action='store_true',
                        help='Output text files')
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = pathlib.Path('ASK')
    assert data_dir.is_dir()

    input_dir = data_dir / 'xml'
    assert input_dir.is_dir()
    output_dir = data_dir / 'txt'

    if not output_dir.is_dir():
        output_dir.mkdir()

    metadata_dict = {
        "lang": [],
        "cefr": []
    }

    for input_file in input_dir.iterdir():
        output_file = str(output_dir / input_file.stem) + '.txt'

        tree = ET.parse(input_file.open())
        root = tree.getroot()

        metadata = root.find('./teiHeader/profileDesc/particDesc/person')
        assert metadata, "Could not find metadata in file " + str(input_file)

        language_node = metadata.find("./p[@n='language']")
        cefr_score_node = metadata.find("./p[@n='CEFRscore']")
        language = language_node.text if language_node is not None else 'N/A'
        cefr_score = cefr_score_node.text if cefr_score_node is not None else 'N/A'

        if language != 'N/A' and cefr_score != 'N/A':
            metadata_dict['lang'].append(language)
            metadata_dict['cefr'].append(cefr_score)

        text = root.find('text')
        assert text, "Missing text in file " + str(input_file)

        if not args.output:
            continue

        with open(output_file, 'w') as outfile:
            outfile.write('# language: %s\n' % language)
            outfile.write('# CEFRscore: %s\n' % cefr_score)

            for paragraph in text.iter('p'):
                for sentence in paragraph.iter('s'):
                    words = [w.text for w in sentence.iter('word') if w.text]
                    joined = ' '.join(words)
                    joined = re.sub(r'\s([.,?!:;])', r'\1', joined)
                    outfile.write(joined + '\n')
                outfile.write('\n')
        print('Wrote ' + output_file)

    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.to_csv('metadata.csv')
    cefr_by_lang = metadata_df.groupby(['lang', 'cefr']).size().unstack()

    print(cefr_by_lang.fillna(0))
    cefr_by_lang.plot.bar()


if __name__ == '__main__':
    main()
