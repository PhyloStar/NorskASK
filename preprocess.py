import argparse
import pathlib
from xml.etree import ElementTree

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
        'lang': [],
        'cefr': [],
        'testlevel': [],
        'age': [],
        'gender': [],
        'topic': [],
        'num_tokens': [],
        'title': []
    }

    for input_file in input_dir.iterdir():
        if input_file.suffix != '.xml':
            print('Skipping ' + str(input_file))
            continue

        try:
            tree = ElementTree.parse(input_file.open())
        except Exception:
            print('Unable to parse ' + str(input_file))
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

        metadata_dict['lang'].append(language)
        metadata_dict['cefr'].append(cefr_score)
        metadata_dict['testlevel'].append(testlevel)
        metadata_dict['age'].append(age)
        metadata_dict['gender'].append(gender)
        metadata_dict['topic'].append(topic)

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
            with open(output_file, 'w') as outfile:
                outfile.write('# language: %s\n' % language)
                outfile.write('# CEFRscore: %s\n' % cefr_score)

                body = text.find('body')

                head = body.find('.//head')
                if head is not None:
                    outfile.write('\n')
                    for sentence in head.iter('s'):
                        words = [w.text for w in sentence.iter('word') if w.text]
                        num_tokens += len(words)
                        joined = ' '.join(words)
                        outfile.write(joined + '\n')
                outfile.write('\n')

                for paragraph in body.iter('p'):
                    for sentence in paragraph.iter('s'):
                        words = [w.text for w in sentence.iter('word') if w.text]
                        num_tokens += len(words)
                        joined = ' '.join(words)
                        outfile.write(joined + '\n')
                    outfile.write('\n')
        metadata_dict['num_tokens'].append(num_tokens)

    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.to_csv('metadata.csv', index=False)
    cefr_by_lang = metadata_df.groupby(['lang', 'cefr']).size().unstack(fill_value=0)

    print(cefr_by_lang.fillna(0))
    cefr_by_lang.plot.bar()


if __name__ == '__main__':
    main()
