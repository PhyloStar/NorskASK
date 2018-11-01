import pathlib
import subprocess


def main():
    data_dir = pathlib.Path('ASK')
    assert data_dir.is_dir()

    input_dir = data_dir / 'txt'
    assert input_dir.is_dir()
    output_dir = data_dir / 'conll'

    if not output_dir.is_dir():
        output_dir.mkdir()

    for input_file in input_dir.iterdir():
        if input_file.suffix != '.txt':
            print('Skipping ' + str(input_file))
            continue

        output_file = (output_dir / input_file.stem).with_suffix('.conll')
        print(str(output_file))
        cmd = ['./rest-udpipe', str(input_file), str(output_file)]
        try:
            subprocess.run(cmd).check_returncode()
        except subprocess.CalledProcessError as e:
            print(str(e))


if __name__ == '__main__':
    main()
