Raw Reference Performance Test measurements from LFP graphite cells are available in `/app/data`.

The named C1 and C2 cell folders contain the development data. `/app/data/heldout` contains 255 samples from unseen C3 cells. The original cell names and RPT order are not shown. Each held out sample must be predicted from its battery measurements only.

For development cell i at RPT j, Q_ref(i,j) is the C/3 reference discharge capacity from rows where `Segment = ref_dchg`. SOH percent is Q_ref(i,j) divided by Q_ref(i,0), then multiplied by 100. If more than one reference discharge is present, use the one with the largest capacity. RPT 0 is the beginning of life reference for that cell.

Build the model using diagnostic signals that do not use capacity. `Capacity(Ah)`, `Energy(Wh)`, SOH, RPT number, cycle count, date, time, Phase and cell identity must not be used as predictors. During development validation, all observations from the same cell must stay together.

Write `/app/results/predictions.csv` with `sample_id` and `predicted_SOH_pct` for every held out sample.

Write `/app/results/metrics.json` with `final_model`, `selected_features`, `development_cells`, `validation_MAE`, `validation_RMSE` and `validation_R2`. These validation values must come from development cells only.

Write `/app/results/feature_ranking.csv` with a `feature` column and at least one numeric ranking or importance score.

Also write the report to `/app/results/report.md` and figures to `/app/results/figures/`.
