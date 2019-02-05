import pickle
import subprocess
from typing import Any, Dict

from masterthesis.utils import RESULTS_DIR

GIT_CMD = ['git', 'rev-parse', '--verify', 'HEAD']


class Results:
    def __init__(self, script_name, config, history, true, predictions, git_rev):
        self.script_name = script_name
        self.config = config
        self.history = history
        self.predictions = predictions
        self.true = true
        self.git_revision = git_rev


def save_results(script_name: str,
                 config: Dict[str, Any],
                 history,
                 true,
                 predictions):
    try:
        result = subprocess.run(GIT_CMD, stdout=subprocess.PIPE)
        git_rev = result.stdout.decode().strip()
    except FileNotFoundError:
        print('ERROR: Could not execute Git')
        git_rev = 'unknown'

    if not RESULTS_DIR.is_dir():
        RESULTS_DIR.mkdir()

    results_file = RESULTS_DIR / (script_name + '.pkl')
    print(results_file)
    results_obj = Results(script_name, config, history, true, predictions, git_rev)

    pickle.dump(results_obj, results_file.open('wb'))
