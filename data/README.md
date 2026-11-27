# Input data for Study 1, design v3

| Path                                    | Committed | What |
| --------------------------------------- | --------- | ---- |
| `raw/tej/`                              | no        | Licensed TEJ daily stock data (see `raw/tej/README.md`); `calibration.tej_path` |
| `investor_statistics.csv`               | yes       | TDCC "Investors with Trading Accounts by Age" table as downloaded; `calibration.tdcc_age_path`. Gives the two age levels |
| `attribute_data.csv`                    | yes       | Further hand-collected statistics: `attribute, statistic, value, year, source_url`. Copied into `results/calibration/attributes.json` as sources |
| `benchmarks/barber_2007_table3_panelB.csv` | yes    | Barber et al. (2007), Table 3 Panel B (PGR, PLR, ratio by investor type) |
| `sources/`                              | yes       | Optional saved copies of cited pages |

`calibrate` reads these and writes `results/calibration/`. The price generator and the persona
builder read only those JSON files.

**Age levels.** `younger_age` / `older_age` are the median ages of individual investors aged 40
and under / 41 and over, interpolated inside the TDCC brackets (0–19, 20–30, 31–40, 41–50,
51–60, 61+); institutions are excluded. Put the TDCC page URL in
`calibration.tdcc_source_url` so it is recorded with the values.

**Optional rows for `attribute_data.csv`** (shown in the viewer, not used in prompts this
round), for example `trading_activity,turnover_individuals_low,3.08,1995-1999,<Barber 2007>`
or FSC survey figures for `financial_literacy` / `experience`.
