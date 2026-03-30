"""
PyxTech Data Loader
===================
Loads pipeline output data for the dashboard.

Current mode: CSV files from /data folder (GitHub repo)
Future mode:  PostgreSQL / oceandb (flip USE_SQL = True)

To switch to SQL later:
    1. Set USE_SQL = True
    2. Add DB credentials to .streamlit/secrets.toml
    3. No other changes needed — same DataStore interface
"""

import os
import re as _re
import pandas as pd
import numpy as np
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
USE_SQL   = False   # Flip to True when oceandb is ready
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data1")
SEGMENTS  = ["GPU", "RAM", "Processor", "Storage", "NetworkAdapter"]

# ── Ground-truth signals (March 2026 sprint) ─────────────────────────────────
CURRENT_SIGNALS = {
    "ea2_major_gpu_launch":           1.0,
    "ea2_datacenter_expansion":       1.0,
    "ea2_major_ai_model_release":     1.0,
    "ea2_hyperscaler_capex_increase": 1.0,
    "ea2_ai_demand_index":            0.07,
    "ea1_dram_shortage_active":       1.0,
    "ea1_price_spike_detected":       1.0,
    "ea1_used_market_active":         1.0,
    "ea1_supply_pressure_index":      0.22,
    "ea3_new_export_control":         1.0,
    "ea3_new_tariff_action":          1.0,
    "ea3_policy_risk_index":          0.10,
    "combined_market_pressure":       0.14,
}


def _gpu_tier(row):
    cap = str(row.get("capacity", ""))
    m   = _re.search(r"(\d+)\s*GB", cap, _re.IGNORECASE)
    gb  = int(m.group(1)) if m else 0
    t   = str(row.get("type", "")).upper()
    if ("GDDR6" in t or "HBM" in t) and gb >= 16: return "AI / Datacenter"
    if gb >= 8 and ("GDDR6" in t or "GDDR5" in t): return "Compute / Workstation"
    return "Legacy / Consumer"


def _read_csv(filename, fallback=None):
    """Read a CSV from /data, return empty DataFrame if not found."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    if fallback:
        path2 = os.path.join(DATA_DIR, fallback)
        if os.path.exists(path2):
            return pd.read_csv(path2, low_memory=False)
    return pd.DataFrame()


def _read_xlsb(filename):
    """Read an .xlsb file from /data1 folder."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        from pyxlsb import open_workbook
        rows = []
        with open_workbook(path) as wb:
            with wb.get_sheet(1) as sheet:
                for i, row in enumerate(sheet.rows()):
                    rows.append([item.v for item in row])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows[1:], columns=rows[0])
    except Exception as e:
        print(f"  WARNING reading xlsb {filename}: {e}")
        return pd.DataFrame()


def _read_sql(query):
    """Future: query oceandb / PostgreSQL."""
    try:
        import sqlalchemy
        conn_str = (
            f"postgresql://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}"
            f"@{st.secrets['DB_HOST']}:{st.secrets['DB_PORT']}/{st.secrets['DB_NAME']}"
        )
        engine = sqlalchemy.create_engine(conn_str)
        return pd.read_sql(query, engine)
    except Exception as e:
        st.warning(f"SQL connection failed: {e}. Falling back to CSV.")
        return pd.DataFrame()


# ── Public interface ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_all():
    """
    Load all data needed by the dashboard.
    Returns a dict of DataFrames.
    Cached for 5 minutes — auto-reloads when data files update.
    """
    d = {}

    if USE_SQL:
        # ── Future SQL queries ────────────────────────────────────────────
        d["momentum"]   = _read_sql("SELECT * FROM gc_id_momentum")
        d["category"]   = _read_sql("SELECT * FROM category_summary_stats")
        d["metrics"]    = _read_sql("SELECT * FROM metrics_all_models")
        d["causal"]     = _read_sql("SELECT * FROM causal_analysis_per_gc_id")
        d["forecasts"]  = _read_sql("SELECT * FROM phase3_future_forecasts")
        d["feature_matrix"] = _read_sql("SELECT * FROM feature_matrix")
        d["per_gc_metrics"] = _read_sql("SELECT * FROM per_gc_id_metrics")

    else:
        # ── CSV mode (current) ────────────────────────────────────────────
        d["momentum"]   = _read_csv("gc_id_momentum.csv")
        d["category"]   = _read_csv("category_summary_stats.csv")
        d["metrics"]    = _read_csv("metrics_all_models.csv")
        d["causal"]     = _read_csv("causal_analysis_per_gc_id.csv",
                                     "causal_analysis_per_gc_id__1_.csv")
        d["per_gc_metrics"] = _read_csv("per_gc_id_metrics_summary.csv")

        # Feature matrices — concatenate all segments
        fm_parts = []
        for seg in SEGMENTS:
            # Try CSV first, then xlsb (Processor may be xlsb to reduce file size)
            fm = _read_csv(f"feature_matrix_{seg}.csv")
            if fm.empty:
                fm = _read_xlsb(f"feature_matrix_{seg}.xlsb")
            if not fm.empty:
                fm["segment"] = seg
                fm_parts.append(fm)
        d["feature_matrix"] = pd.concat(fm_parts, ignore_index=True) if fm_parts else pd.DataFrame()

        # Forecasts — concatenate all segments
        fc_parts = []
        for seg in SEGMENTS:
            fc = _read_csv(f"phase3_future_forecast_{seg}.csv")
            if not fc.empty:
                fc["segment"] = seg
                fc_parts.append(fc)
        d["forecasts"] = pd.concat(fc_parts, ignore_index=True) if fc_parts else pd.DataFrame()

    # ── Feature vector (signal state) ────────────────────────────────────
    fv_df   = _read_csv("enterprise_feature_vector.csv")
    raw_fv  = fv_df.iloc[0] if not fv_df.empty else pd.Series(dtype=float)
    fv      = raw_fv.copy()
    for k, v in CURRENT_SIGNALS.items():
        if k not in fv.index or float(fv.get(k, 0)) == 0:
            fv[k] = v
    d["fv"] = fv

    # ── Enrich momentum with GPU tier ─────────────────────────────────────
    if not d["momentum"].empty:
        d["momentum"]["gpu_tier"] = d["momentum"].apply(_gpu_tier, axis=1)

    # ── Use feature matrix as historical data source ──────────────────────
    # Feature matrix has full 50-month history per GC_ID — richer than augmented
    if not d["feature_matrix"].empty and "fmv" in d["feature_matrix"].columns:
        d["historical"] = d["feature_matrix"][
            ["month", "gc_id", "fmv", "segment",
             "manufacturer", "model_name", "model_number",
             "condition", "region"]
            + [c for c in d["feature_matrix"].columns
               if c.startswith("ea") or c.startswith("fmv_lag")]
        ].copy()
    else:
        d["historical"] = pd.DataFrame()

    return d


def get_best_model(metrics_df, segment):
    """Return the best performing model for a segment based on T+4 test MdAPE."""
    if metrics_df.empty or "segment" not in metrics_df.columns:
        return "XGBoost"
    seg = metrics_df[metrics_df["segment"] == segment].copy()
    if "split"   in seg.columns: seg = seg[seg["split"]   == "test"]
    if "horizon" in seg.columns: seg = seg[seg["horizon"] == "T+4"]
    if seg.empty or "mape_median" not in seg.columns:
        return "XGBoost"
    return seg.loc[seg["mape_median"].idxmin(), "model"]


def get_gc_history(historical_df, gc_id):
    """Get the full price history for a single GC_ID, sorted by month."""
    if historical_df.empty or "gc_id" not in historical_df.columns:
        return pd.DataFrame()
    hist = historical_df[historical_df["gc_id"].astype(str) == str(gc_id)].copy()
    if "month" in hist.columns:
        hist["month"] = pd.to_datetime(hist["month"])
        hist = hist.sort_values("month").reset_index(drop=True)
    return hist


def get_gc_forecast(forecasts_df, gc_id, model=None):
    """Get the 4-month forecast for a single GC_ID."""
    if forecasts_df.empty or "gc_id" not in forecasts_df.columns:
        return pd.DataFrame()
    fc = forecasts_df[forecasts_df["gc_id"].astype(str) == str(gc_id)].copy()
    if model and "model" in fc.columns:
        fc = fc[fc["model"] == model]
    h_order = {"T+1": 1, "T+2": 2, "T+3": 3, "T+4": 4}
    if "horizon" in fc.columns:
        fc["h_num"] = fc["horizon"].map(h_order)
        fc = fc.sort_values("h_num")
    return fc


def get_per_gc_accuracy(per_gc_df, gc_id):
    """Get per-GC_ID model accuracy stats."""
    if per_gc_df.empty or "gc_id" not in per_gc_df.columns:
        return pd.DataFrame()
    return per_gc_df[per_gc_df["gc_id"].astype(str) == str(gc_id)]
