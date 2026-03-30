# PyxTech AI Centre of Excellence — Dashboard

Enterprise hardware market intelligence and Pyx FMV forecasting dashboard.

## Architecture

```
/
├── pyxtech_app.py          # Main Streamlit dashboard
├── data_loader.py          # Data abstraction layer (CSV → SQL-ready)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   ├── config.toml         # Theme and server config
│   └── secrets.toml        # API keys (NOT committed — set in Streamlit Cloud)
└── data/                   # Pipeline output CSVs (updated weekly by automation)
    ├── gc_id_momentum.csv
    ├── category_summary_stats.csv
    ├── metrics_all_models.csv
    ├── per_gc_id_metrics_summary.csv
    ├── causal_analysis_per_gc_id.csv
    ├── enterprise_feature_vector.csv
    ├── feature_matrix_GPU.csv
    ├── feature_matrix_RAM.csv
    ├── feature_matrix_Processor.csv
    ├── feature_matrix_Storage.csv
    ├── feature_matrix_NetworkAdapter.csv
    ├── phase3_future_forecast_GPU.csv
    ├── phase3_future_forecast_RAM.csv
    ├── phase3_future_forecast_Processor.csv
    ├── phase3_future_forecast_Storage.csv
    └── phase3_future_forecast_NetworkAdapter.csv
```

## Segments

| Segment | GC_IDs | Model |
|---------|--------|-------|
| GPU | ~95 | XGBoost / LightGBM / LSTM |
| RAM | ~95 | XGBoost / LightGBM / LSTM |
| Processor | ~1,216 | XGBoost / LightGBM / LSTM |
| Storage | ~1,049 | XGBoost / LightGBM / LSTM |
| Network Adapter | ~9 | XGBoost / LightGBM / LSTM |

## Deployment

### Streamlit Community Cloud
1. Fork / push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `pyxtech_app.py`
4. Add secrets in App Settings:
   ```
   OPENAI_API_KEY = "sk-proj-..."
   ```
5. Deploy

### Local development
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-proj-...
streamlit run pyxtech_app.py
```

## Weekly Data Update (Automated)
The GitHub Actions workflow (`.github/workflows/weekly_update.yml`) runs every Monday at 6am IST:
1. Triggers the Colab pipeline via Google Colab API
2. Pulls updated CSVs from Google Drive
3. Commits them to `/data` folder
4. Streamlit Cloud auto-reloads on new commit

## Data Layer
`data_loader.py` provides a clean abstraction:
- **Current mode**: reads from `/data` CSV files
- **Future mode**: flip `USE_SQL = True` to query PostgreSQL / oceandb directly

## Tech Stack
- **Forecasting**: XGBoost (quantile regression), LightGBM (quantile regression), LSTM (MC Dropout)
- **Market Intelligence**: EA1 (supply chain, 11 feeds), EA2 (AI technology, 22 feeds), EA3 (policy & trade, 17 feeds)
- **Dashboard**: Streamlit + Matplotlib + GPT-4o (PyxieAnalyst)
- **Pipeline**: Google Colab → Google Drive → GitHub → Streamlit Cloud
