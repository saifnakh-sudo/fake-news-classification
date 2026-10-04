# Fake News Classification & Analysis

**A Python data science project developed during GDG on Campus – ENSAK.**

This project explores how a machine-learning workflow can classify workshop news examples and reveal patterns in their text and sharing activity. It combines data preparation, feature engineering, Random Forest classification, visual evaluation, and exploratory text clustering.

**Scope:** an educational prototype on template-style teaching data. It does not independently verify facts or establish whether a real article is trustworthy.

![Dataset labels and article lengths](results/dataset_overview.png)

## Explore the project

- **[Walkthrough notebook](notebooks/fake_news_analysis.ipynb)** — data inspection, model evaluation, charts, and limitations.
- **[Runnable analysis](src/analysis.py)** — generates all saved charts and metrics.
- **[Measured results](results/metrics.json)** — dataset fingerprint, split, scores, baselines, and library versions.
- **[Original submissions](originals/)** — the original workshop scripts and notebook, preserved separately.

## What the workflow does

1. Loads and validates the CSV schema; handles missing text and removes exact duplicate rows.
2. Explores class balance, body lengths, and sharing activity.
3. Creates numerical features from headline patterns, text length, and share counts.
4. Trains a 100-tree Random Forest on a stratified 80/20 split with seed 42. The share-clipping threshold is calculated from training data only.
5. Evaluates accuracy, fake-class F1, and a confusion matrix, alongside a majority-class baseline.
6. Examines feature importance, headline keywords, short-text/high-sharing patterns, and TF-IDF / K-Means text groups.
7. Runs a separate exploratory share-count regression using only text statistics, compared with a median baseline.

The classifier uses **numeric style and sharing features**. TF-IDF is used for exploratory clustering, not for the classifier. It does not query outside sources or perform factual verification.

## Results and interpretation

| Experiment | Recorded holdout result | Comparison |
| --- | --- | --- |
| News classification | Accuracy 100.0%; fake-class F1 1.000 | Majority baseline accuracy 59.4% |
| Sharing regression | MAE 12,823.09 shares | Median baseline MAE 8,001.54 shares |

**Interpretation:** classification separates the repeated teaching templates; this is not real-world accuracy. The sharing regressor performs worse than the median baseline on this split.

See [the evaluation report](results/classification_report.txt) for the actual run and [metrics.json](results/metrics.json) for exact scores and environment details.

![Classification confusion matrix](results/confusion_matrix.png)

![Random Forest feature importance](results/feature_importance.png)

The supplied 800-row dataset contains 475 `Real` and 325 `Fake` labels, but only **10 distinct article bodies**. Its repeated wording and template-style headlines strongly suggest constructed teaching examples; the generation method and independent label verification are not documented. Repeated bodies appear in both sides of the random split. A high score here reflects separation within this dataset, not proven performance on real news.

Source/domain strings are not evidence that an article came from the named outlet. The sharing-pattern heuristic does not detect actual bots, and the exploratory regression is not a validated prediction of future virality.

## Run locally

Python **3.11** was used for the recorded run. Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS / Linux:

```bash
source .venv/bin/activate
```

Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python src/analysis.py
```

Charts and reports are written to `results/`. There are no interactive plot windows. To compare the older filtered CSV without replacing the default results:

```bash
python src/analysis.py --data data/fake_news_no_outliers.csv --output .local/filtered-results
```

Open `notebooks/fake_news_analysis.ipynb` in VS Code with a Python/Jupyter kernel from this environment to step through the project.

Run the checks for the corrected logic:

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
data/          Supplied CSVs, schema, and provenance notes
notebooks/     Executed portfolio walkthrough
originals/     Original workshop code and notebook
results/       Generated charts, metrics, and evaluation report
src/           Reproducible portfolio analysis
tests/         Checks for uppercase scoring, clipping, and regression leakage
```

## Changes in the portfolio edition

The workshop submission and this edition are intentionally distinguished:

- Corrected the original uppercase headline test, which ran after lowercasing the text.
- Used the actual CSV column names and added the missing notebook data-loading step.
- Removed the unused NLTK dependency from the runnable edition.
- Added input validation and fitted share clipping only on the classification training set.
- Removed share-derived features from share-count regression to avoid directly exposing its target.
- Used only as many text clusters as there are distinct TF-IDF vectors, up to four.
- Added deterministic outputs, comparison baselines, saved charts, and explicit limitations.

These are preparation changes made for the portfolio edition; they should not be read as claims about what was implemented in the original workshop.

## Next improvements

- Evaluate on a documented corpus of authentic, independently labeled articles.
- Use source, time, or text-family holdouts to reduce overlap between training and test data.
- Compare text-only baselines and inspect failure cases.
- Establish dataset provenance and redistribution terms before using a new corpus.

## Project context and attribution

Ahmed Saifeddine Nakhli participated in this project at **GDG on Campus – ENSAK**. The certificate is a local chapter's project-completion achievement, not a Google professional certification. The workshop files do not identify every contributor or individual contribution; no sole-authorship claim is made.

This repository preserves the supplied educational materials and adds a documented portfolio edition. No open-source license is assigned because the supplied files do not establish the authorship and licensing of all workshop materials. See [data provenance](data/README.md).
