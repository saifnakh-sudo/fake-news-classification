# Data

These are the two CSV files I used in the project. The main analysis uses `fake_news.csv` by default.

| File | Articles | Real | Fake |
| --- | ---: | ---: | ---: |
| `fake_news.csv` | 800 | 475 | 325 |
| `fake_news_no_outliers.csv` | 717 | 470 | 247 |

## Columns

| Column | Description |
| --- | --- |
| `ID_Article` | Article identifier |
| `Title` | Headline |
| `Corps_Textee` | Article body |
| `URL_Source` | Source or domain string |
| `Nb_Partages` | Number of shares |
| `Etiquette` | Dataset label: `Real` or `Fake` |

The main dataset has 10 distinct article bodies across 800 rows. Repeated text appears in both the training and test sets, so the evaluation results are specific to this dataset.
