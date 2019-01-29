import argparse
from collections import defaultdict
from xml.etree import ElementTree

import pandas as pd

from masterthesis.utils import PROJECT_ROOT, DATA_DIR


test_topics = {
    'geografi norge folk ', 'innvandring ', 'innvandring politikk valg ', 'idrett/sport ',
    'bolig geografi ', 'arbeid yrke ', 'økonomi holdning ', 'humor kultur ',
    'politikk norge holdning ', 'litteratur bok ', 'familie befolkning norge ',
    'litteratur dikt idrett ', 'folk utdannelse ', 'politikk holdning ', 'media tv ',
    'religion ', 'helse organ ', 'folk følelser '
}

dev_topics = {
    'helse ', 'helse arbeid innvandring ', 'litteratur dikt språk ', 'organisasjon ',
    'helse røyking ', 'barn familie ', 'økonomi ', 'opplevelse ', 'familie flytting ',
    'eldre familie ', 'barn idrett/sport ', 'litteratur dikt venner ', 'arbeid innvandring ',
    'utdannelse språk ', 'idrett/sport kultur ', 'holdning '
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', action='store_true',
                        help='Output text files')
    return parser.parse_args()


def main():
    args = parse_args()

    assert DATA_DIR.is_dir()

    input_dir = DATA_DIR / 'xml'
    assert input_dir.is_dir()
    output_dir = DATA_DIR / 'txt'

    if not output_dir.is_dir():
        output_dir.mkdir()

    metadata_dict = defaultdict(list)

    for input_file in input_dir.iterdir():
        if input_file.suffix != '.xml':
            print('Skipping ' + str(input_file))
            continue

        try:
            tree = ElementTree.parse(input_file.open(encoding='utf8'))
        except Exception as e:
            print('Unable to parse ' + str(input_file))
            print(str(e))
            raise
        root = tree.getroot()

        metadata_node = root.find('./teiHeader/profileDesc/particDesc/person')
        assert metadata_node, "Could not find metadata in file " + str(input_file)

        language_node = metadata_node.find("./p[@n='language']")
        cefr_score_node = metadata_node.find("./p[@n='CEFRscore']")
        testlevel_node = metadata_node.find("./p[@n='testlevel']")
        age_node = metadata_node.find("./p[@n='age']")
        gender_node = metadata_node.find("./p[@n='gender']")
        topic_node = metadata_node.find("./p[@n='tema']")

        language = language_node.text if language_node is not None else 'N/A'
        cefr_score = cefr_score_node.text if cefr_score_node is not None else 'N/A'
        testlevel = testlevel_node.text if testlevel_node is not None else 'N/A'
        try:
            age = int(age_node.text) if age_node is not None else -1
        except ValueError:
            age = -1
        gender = gender_node.text if gender_node is not None else 'N/A'
        topic = topic_node.text if topic_node is not None else 'N/A'

        if cefr_score == 'N/A':
            split = 'N/A'
        elif topic in test_topics:
            split = 'test'
        elif topic in dev_topics:
            split = 'dev'
        else:
            split = 'train'

        metadata_dict['lang'].append(language)
        metadata_dict['cefr'].append(cefr_score)
        metadata_dict['testlevel'].append(testlevel)
        metadata_dict['age'].append(age)
        metadata_dict['gender'].append(gender)
        metadata_dict['topic'].append(topic)
        metadata_dict['split'].append(split)
        metadata_dict['filename'].append(input_file.stem)

        text = root.find('text')
        assert text, "Missing text in file " + str(input_file)

        title_node = text.find("./front/div[@type='title']")
        title = title_node.text if topic_node is not None else 'N/A'

        metadata_dict['title'].append(title)

        if not args.output:
            num_tokens = sum(1 for _ in text.iter('word'))
        else:
            num_tokens = 0
            output_file = str(output_dir / input_file.stem) + '.txt'
            with open(output_file, 'w', encoding='utf-8') as outfile:
                body = text.find('body')

                head = body.find('.//head')
                if head is not None:
                    for sentence in head.iter('s'):
                        words = [w.text for w in sentence.iter('word') if w.text]
                        num_tokens += len(words)
                        joined = ' '.join(words)
                        outfile.write(joined + '\n')

                for paragraph in body.iter('p'):
                    outfile.write('\n')
                    for sentence in paragraph.iter('s'):
                        words = [w.text for w in sentence.iter('word') if w.text]
                        num_tokens += len(words)
                        joined = ' '.join(words)
                        outfile.write(joined + '\n')
        metadata_dict['num_tokens'].append(num_tokens)

    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.sort_values('filename').to_csv(DATA_DIR / 'metadata.csv', index=False)
    cefr_by_lang = metadata_df.groupby(['lang', 'cefr']).size().unstack(fill_value=0)

    print(cefr_by_lang.fillna(0))
    cefr_by_lang.plot.bar()


if __name__ == '__main__':
    main()
