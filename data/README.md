# Workshop data

These are the CSV files supplied with Ahmed Saifeddine Nakhli's GDG on Campus – ENSAK project materials.

| File | Rows | Real | Fake | Use |
| --- | ---: | ---: | ---: | --- |
| `fake_news.csv` | 800 | 475 | 325 | Default analysis input |
| `fake_news_no_outliers.csv` | 717 | 470 | 247 | Earlier filtered version, retained for comparison |

The exact filtering procedure for the second file was not included, so the portfolio edition uses the original 800-row file by default.

## Schema

| Column | Meaning |
| --- | --- |
| `ID_Article` | Article identifier |
| `Title` | Headline |
| `Corps_Textee` | Body text (original column spelling) |
| `URL_Source` | Source/domain string |
| `Nb_Partages` | Number of shares |
| `Etiquette` | Workshop label: `Real` or `Fake` |

## Interpretation and provenance

The 800-row file has only 10 distinct article bodies, with repeated generic wording and template-style titles. This strongly suggests constructed teaching data; the supplied files do not document the generator, collection method, label verification, or an upstream license. Domain names in the source column do not establish that articles were published by those outlets.

Use the data to understand the workflow. Do not treat it as a verified corpus of real news, or interpret a high score on it as evidence of reliable fact checking. No separate upstream dataset license was provided.
