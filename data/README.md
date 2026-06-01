# Data

Place the three source files here (or set `ADV_DATA_DIR` to their directory).
These files are **not redistributed** in this repository.

| File | Description | Key columns (high level) |
|---|---|---|
| `commercial_registry_monthly_clean_for_simulation.csv` | Monthly commercial-registry counts | a monthly date column; counts of company divisions (incorporation- and absorption-type), split-related capital increases/decreases, special liquidations, bankruptcies or civil rehabilitations, corporate reorganizations, dissolutions, incorporations, merger-related incorporations/dissolutions, and total registrations |
| `マクロ指標.csv` | Macroeconomic indicators | a monthly date column; stock indices (e.g., Nikkei, TOPIX), exchange rate, lending rate |
| `貸出債券市場取引動向_全銀協_.csv` | Loan / bond-market conditions (Japanese Bankers Association) | a monthly date column; syndicated-loan amounts and related series |

The panel spans January 2009 – January 2026 (205 months). Column names are resolved in
`src/abusive_division_simulation_v2.py` (`prepare_monthly_panel`); adjust the rename map
there if your column headers differ.
