# Contributing

Thanks for helping improve Neural Lab.

## Local setup

Use Python 3.11 if possible, create a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Before opening a pull request

Run:

```bash
python -m pytest
streamlit run app.py
```

For changes to model inputs, outputs, preprocessing, checkpoint formats, or device selection, add or update tests. Keep dataset downloads out of unit tests. Do not commit downloaded datasets, checkpoints, credentials, or local environment files.

## Pull requests

- Explain the behavior change and why it is needed.
- Include tests for bug fixes and new behavior.
- Report which commands you actually ran and any checks that remain unverified.
