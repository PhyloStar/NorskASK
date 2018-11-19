import datetime as dt
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Any

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
GIT_CMD = ['git', 'rev-parse', '--verify', 'HEAD']


class Results:
    def __init__(self, script_name, config, history, predictions, git_rev):
        self.script_name = script_name
        self.config = config
        self.history = history
        self.predictions = predictions
        self.git_revision = git_rev


def save_results(script_name: str,
                 config: Dict[str, Any],
                 history,
                 predictions):
    timestamp = dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    result = subprocess.run(GIT_CMD, stdout=subprocess.PIPE)
    git_rev = result.stdout.decode().strip()
    if not RESULTS_DIR.is_dir():
        RESULTS_DIR.mkdir()

    results_file = RESULTS_DIR / (script_name + '-' + timestamp + '.pkl')
    print(results_file)
    results_obj = Results(script_name, config, history, predictions, git_rev)

    pickle.dump(results_obj, results_file.open('wb'))
