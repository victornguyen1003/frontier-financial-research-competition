# TEJ daily stock data (licensed, git-ignored)

Put the TEJ export here. Everything in this folder except this README is ignored by git.

Expected file: `fulldata_1995to1999.csv` (path set in `calibration.tej_path`, `config/study1.yaml`).
If you have several files (e.g. one per year), drop them all here; the loader will read every
`*.csv` in this folder unless `calibration.tej_path` names one file.

Required columns (Chinese headers, as exported):

| Column       | Used for                                   |
| ------------ | ------------------------------------------ |
| 公司簡稱     | stock identifier                           |
| 年月日       | trading date                               |
| 收盤價(元)   | close → month-end closes, monthly log returns |
| 報酬率-Ln    | daily log return → random-walk tests       |
| 市場別       | filter `TSE`                               |
| TSE 產業別   | filter `!= "-"` (drops funds/ETFs)         |
| 漲跌停       | price-limit share (descriptive)            |

Please note the file encoding (UTF-8 / Big5 / UTF-8-BOM) and whether `報酬率-Ln` is in percent.
Coverage wanted: 1995-01 to 1999-12 (Barber et al. 2007 sample). 1999 alone works but is weaker.
