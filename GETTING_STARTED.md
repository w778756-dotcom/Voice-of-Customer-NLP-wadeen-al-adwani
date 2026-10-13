# Getting started / التشغيل

## Local or cloud environment (Python 3.12)

```bash
git clone https://github.com/w778756-dotcom/Voice-of-Customer-NLP-wadeen-al-adwani.git
cd Voice-of-Customer-NLP-wadeen-al-adwani
# Install uv from its official instructions if unavailable: https://docs.astral.sh/uv/
bash scripts/run_all.sh
UV_CACHE_DIR=/workspace/.cache/uv uv run --frozen python app.py
```

`run_all.sh` installs locked packages, trains both models, evaluates the fixed test set and runs tests. It stops on failure. On a personal machine outside `/workspace`, set `UV_CACHE_DIR` and `XDG_CACHE_HOME` to writable cache directories before running it. No API keys, account secrets or GPU are required. Package downloads need PyPI; analysis is local.

Open the address printed by Gradio **on your own machine**. In a cloud onboarding task the localhost address is used only for internal smoke checks; no public preview is provided. To use another port, set `PORT=7861`. Stop the server with Ctrl-C. Retrain with `uv run --frozen python scripts/train.py` if `artifacts/model.pkl` is absent. Models are generated locally and deliberately excluded from Git.

## Demo

Paste one comment per line, click **تحليل التعليقات**, then inspect sentiments, review flags, clusters, representative quotes and JSON. Or upload `sample_outputs/demo_input.csv` (UTF-8, column `text`). Limits: 1–1000 nonempty comments, at most 4000 characters/comment, CSV ≤5 MB. Inputs containing personal data are not appropriate for this prototype.

## Notebook / Colab

Open `bayan_NLP_Wadeen_Khalid_Al_adwani.ipynb` or `notebooks/01_exploration.ipynb`. On Colab use a Python 3.12 runtime and run:

```python
!git clone https://github.com/w778756-dotcom/Voice-of-Customer-NLP-wadeen-al-adwani.git
%cd Voice-of-Customer-NLP-wadeen-al-adwani
!pip install -r requirements-day1.txt
!pip install --no-deps -e .
```

Run the notebook cells from the repository root (or `notebooks/`). Restart a Colab runtime if it asks after installation. Local Python 3.12 notebook execution is verified; Colab itself is not remotely verified. For local Jupyter, select `.venv/bin/python` as the kernel; an existing notebook editor can discover it. The project does not depend on a globally installed Jupyter server.

## Rebuild reports

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv run --frozen python scripts/evaluate.py
UV_CACHE_DIR=/workspace/.cache/uv uv run --frozen python scripts/build_docs.py
```

Timing values vary by machine; prediction/F1 outputs are deterministic with the locked environment. Dataset SHA in `reports/training.json` identifies the corpus. Do not alter the CSV then present old metrics as current. Fresh test experiments should use a new untouched corpus.
