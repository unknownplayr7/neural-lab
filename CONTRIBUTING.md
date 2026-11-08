# Contributing

Neural Lab is a macOS-only project. Please develop and test changes on macOS; Apple Silicon testing is preferred for MPS-related changes. Windows and Linux support are out of scope.

Thanks for helping improve Neural Lab.

## Local setup

Use Python 3.11 or newer (Python 3.12 recommended). Use a versioned interpreter command so your virtual environment does not accidentally use an older system Python. The example below uses Python 3.12:

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
python -m pip check
```

If `python3.12` is not installed, install Python 3.12 first or substitute the versioned command for another installed Python 3.11+ interpreter (for example, `python3.11`). The project's `pyproject.toml` requires Python 3.11 or newer.

## Before opening a pull request

Run:

```bash
python -m pytest
python -m streamlit run app.py
```

For changes to model inputs, outputs, preprocessing, checkpoint formats, or device selection, add or update tests. Keep dataset downloads out of unit tests. Do not commit downloaded datasets, checkpoints, credentials, or local environment files.

## Pull requests

- Explain the behavior change and why it is needed.
- Include tests for bug fixes and new behavior.
- Report which commands you actually ran and any checks that remain unverified.
