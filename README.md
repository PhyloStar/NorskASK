# NorskASK
Neural network experiments with Norwegian Learner Language Corpus

# Master thesis on Automated Essay Scoring for Norwegian

The LaTeX source is contanied in the [thesis](./thesis) folder.

## Setup

Create a new virtual environment using Python 3.5 and activate it. Then
install the project in the virtual environment. If you will be doing
development work in the project, install using `pip install -e .[dev]`
instead. Then install the kernel of the virtual environment into Ipython in
order to be able to access it from Jupyter notebooks, even if Jupyter is
installed globally.

```bash
python3.5 -m venv venv
. venv/bin/activate
pip install -e .
python -m ipykernel install --user --name masterthesis \
    --display-name "Python (master thesis)"
```
