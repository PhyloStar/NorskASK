from getpass import getpass
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, IO, Iterable, List

import numpy as np
from sklearn.metrics import f1_score

from masterthesis.utils import RESULTS_DIR


def macrof1(true, pred):
    return f1_score(true, pred, average='macro')


def microf1(true, pred):
    return f1_score(true, pred, average='micro')


def metrics_from_files(fnames: Iterable[str], metrics: Iterable[str]) -> List[float]:
    vals = []  # type: List[float]
    for fname in fnames:
        fpath = RESULTS_DIR / (fname + '.pkl')
        res = pickle.load(fpath.open('rb'))
        true = res.true
        pred = res.predictions
        for metric in metrics:
            if metric == 'macrof1':
                vals.append(macrof1(true, pred))
            if metric == 'microf1':
                vals.append(microf1(true, pred))
    return vals


def make_table_row(line: str, cfg):
    raw_name, _, raw_files = line.partition(':')
    name = raw_name.strip()
    files = raw_files.split()
    assert cfg['models-per-row'] == len(files)
    metrics = cfg['columns-per-model']
    return name, metrics_from_files(files, metrics)


def get_config(items: Iterable[str]) -> Dict[str, Any]:
    cfg = {}  # type: Dict[str, Any]
    for item in items:
        key, raw_val = item.split('=')
        if key == 'models-per-row':
            val = int(raw_val)  # type: Any
        elif key == 'columns-per-model':
            val = raw_val.split(',')
        cfg[key] = val
    return cfg


def make_print_out(names, rows):
    max_value_rows = np.array(rows).argmax(axis=0)
    for row_nr, (name, vals) in enumerate(zip(names, rows)):
        bold = max_value_rows == row_nr
        num_strs = ['$\\mathbf{%.3f}$' % v
                    if b else '$%.3f$' % v
                    for b, v in zip(bold, vals)]
        row = ' & '.join([name] + num_strs) + r' \\'
        print(row)


def process_table(f: IO[str]):
    names = []  # type: List[str]
    rows = []  # type: List[List[float]]
    while True:
        try:
            line = next(f)
        except StopIteration:
            break
        line = line.partition('%')[2].strip()  # Everything after %
        if not line:
            continue
        if line == '$END autotable':
            make_print_out(names, rows)
            print('Found end of table!', file=sys.stderr)
            return
        items = line.split()
        if items[0] == '$META':
            cfg = get_config(items[1:])
        else:
            name, row = make_table_row(line, cfg)
            names.append(name)
            rows.append(row)


def process_file(f: IO[str]):
    while True:
        try:
            line = next(f)
        except StopIteration:
            break
        if line.strip().startswith('% $BEGIN autotable'):
            print('Found start of table!', file=sys.stderr)
            table_name = line.split()[-1]
            if len(sys.argv) > 2 and table_name in sys.argv[2:]:
                process_table(f)
            elif getpass('Process table %s (y/n)? ' % table_name).lower()[0] == 'y':
                process_table(f)
            else:
                print('Skipping table ' + table_name, file=sys.stderr)
    print('process_file finished', file=sys.stderr)


if __name__ == '__main__':
    fname = sys.argv[1]
    process_file(Path(fname).open(encoding='utf-8'))
