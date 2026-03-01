# FlightID — Fare-Inflation Explanation & Corridor Recommendations

This project trains an interpretable model (XGBoost GBM) to predict expected flight fares per-mile and uses SHAP to explain why a fare is inflated. It also produces corridor rankings (cost-effectiveness) and suggests cheaper corridors from the same origin.

Quick start

1. Create a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

2. Run training + explanation pipeline (reads `data/airline_ticket_dataset_csv.csv` by default):

```bash
python src/train_explain.py --csv data/airline_ticket_dataset_csv.csv --out outputs
```

Outputs (written to `outputs/`):
- `model.joblib` — trained model
- `shap_summary.png` — SHAP summary plot
- `inflated_flights.csv` — rows flagged as inflated with suggested alternatives
- `corridor_rankings.csv` — corridors ranked by predicted fare per mile

Files added: `src/train_explain.py`, `src/utils.py`.

See `src/train_explain.py --help` for options.

In order to open the SDSS_Hackathon.pbix file in the DevPost submission, you can either open it on the PowerBI Desktop app or through the [web](app.powerbi.com) and upload it as your own workshop. (Free Trial struggles)
