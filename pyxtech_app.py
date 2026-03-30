"""
PyxTech AI Centre of Excellence — Streamlit Dashboard v3
"""
import os, glob, warnings, re as _re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import io

warnings.filterwarnings("ignore")

# Data source: reads from /data folder (CSV) or SQL — see data_loader.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all as _load_all_from_loader, get_best_model, CURRENT_SIGNALS, SEGMENTS as _SEGS

DRIVE = os.environ.get("PYXTECH_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
SEGMENTS = ["GPU", "RAM", "Processor", "Storage", "NetworkAdapter"]
MODELS   = ["XGBoost", "LightGBM", "LSTM"]
HORIZONS = ["T+1", "T+2", "T+3", "T+4"]

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

FLAGS = {
    "DRAM Shortage Active":        "ea1_dram_shortage_active",
    "Price Spike Detected":        "ea1_price_spike_detected",
    "Used Market Premium":         "ea1_used_market_active",
    "Major GPU Launch":            "ea2_major_gpu_launch",
    "Major AI Model Release":      "ea2_major_ai_model_release",
    "Hyperscaler Capex Surge":     "ea2_hyperscaler_capex_increase",
    "Export Controls Widened":     "ea3_new_export_control",
    "Tariff Action Active":        "ea3_new_tariff_action",
}

SEG_SIGNALS = {
    "GPU":           ["ea2_major_gpu_launch","ea2_ai_demand_index","ea2_datacenter_expansion","ea3_new_export_control"],
    "RAM":           ["ea1_dram_shortage_active","ea1_supply_pressure_index","ea1_price_spike_detected","ea1_used_market_active"],
    "Processor":     ["ea3_new_export_control","ea3_policy_risk_index","ea3_new_tariff_action"],
    "Storage":       ["ea3_policy_risk_index"],
    "NetworkAdapter":["ea2_major_gpu_launch","ea2_datacenter_expansion"],
}

SEG_NARRATIVE = {
    "GPU": {
        "insight": "The GPU market is bifurcated into two completely separate markets. AI/Datacenter-tier (H100, A100, A6000 with GDDR6 ≥16GB) is in active demand at record secondary-market prices. Legacy tier (Quadro K-series, Tesla M/K-series) is in structural freefall. The aggregate Pyx FMV median tells you nothing useful about either tier in isolation.",
        "finding": "NVIDIA Q4 FY2026 revenue: $68.1B (+73% YoY). Blackwell Ultra delivers 50× performance gain — triggering H100/A100 rotation from hyperscaler fleets into secondary market. Used GPU secondary market: avg **$17,087**, peak **$51,826** per unit. 282 EA2 demand signals collected. All 4 GPU-relevant flags ACTIVE.",
        "interpretation": "Blackwell Ultra creates a **rotation, not relief**. When hyperscalers refresh to Blackwell, H100/A100 units enter the secondary market. Enterprise inference workloads absorb them immediately at premium pricing. BIS affiliate rule extension redirects Chinese institutional demand to same secondary pool — compressing supply further.",
        "action": "**AI-tier (H100/A100/A6000): Elevated procurement priority — consider reviewing forward contracts.** Legacy (K-series, Tesla, Quadro K): DO NOT PURCHASE. Never use aggregate GPU Pyx FMV for procurement — use disaggregated tier benchmarks.",
        "signals_active": ["ea2_major_gpu_launch","ea2_datacenter_expansion","ea2_major_ai_model_release","ea3_new_export_control"],
    },
    "RAM": {
        "insight": "RAM is the single most urgent procurement action in this report. The shortage is active, broad-based, and accelerating. EA1 supply signals and live price data are fully converged — both pointing the same direction simultaneously.",
        "finding": "3M median: **+59.5%** | Mean: **+99.5%** | Max: **+593%** in 90 days. DDR4 6M: +56%. DDR5 6M: **+120%**. HBM3E: fully allocated, no spot market. Used enterprise RAM transacting at **$5,910/unit** (secondary premium). TrendForce: *'no clear signs of easing.'* NAND supplier revenue +23.8% QoQ (price-driven, not volume).",
        "interpretation": "When both spot price data and supply chain intelligence signals point upward simultaneously, the increase is structural. The right-skewed distribution (median +59.5%, mean +99.5%) indicates broad-based shortage with explosive outlier movers — not an isolated event. The 3M vs 6M acceleration check confirms the rate of change is **increasing**, not plateauing.",
        "action": "**BUY NOW or lock forward contracts. All RAM types.** Do not wait for April seasonal dip — shortage likely suppresses or eliminates it. DDR5: highest urgency (+120% 6M). HBM3E: engage manufacturer directly for allocation. Update all BOMs costed at pre-shortage rates immediately.",
        "signals_active": ["ea1_dram_shortage_active","ea1_price_spike_detected","ea1_used_market_active","ea1_supply_pressure_index"],
    },
    "Processor": {
        "insight": "The Processor category contains two completely different markets — modern Xeon Scalable and legacy Xeon E5/Pentium/Celeron. The aggregate –5.4% median is a statistical artefact that accurately describes neither market. Never use it for procurement decisions.",
        "finding": "Bimodal distribution: left peak –60% to –80% (legacy crash), right cluster –5% to +30% (Scalable). Intel dominates: 1,089 GC_IDs. Bottom movers: Xeon L5520 (–89.9%), Pentium G3450 (–88.7%). Pyx FMV for legacy has reached **$1–$3/unit** — below the logistics cost of moving the hardware.",
        "interpretation": "No EA1, EA2, or EA3 signals are specific to the Processor category — price dynamics here are **lifecycle-driven**, not shortage or demand-driven. Export controls affect GPU/RAM supply chains but have minimal direct Processor impact. The bifurcation is structural and final-phase.",
        "action": "**Modern Scalable (3rd/4th gen Gold/Platinum): Monitor — stable pricing, review on normal cycles.** Legacy Xeon E5/Pentium/Celeron: DO NOT PURCHASE at any price above scrap. Never use –5.4% aggregate to benchmark a Scalable configuration.",
        "signals_active": ["ea3_new_export_control","ea3_new_tariff_action"],
    },
    "Storage": {
        "insight": "Storage is the calmest segment. No active shortage signals, no policy-driven demand redirects. The market behaves as normal supply-demand equilibrium with statistically significant seasonality (ANOVA p < 0.001).",
        "finding": "3M median: **+2.7%** across 1,049 GC_IDs. October = best buying window (+6% avg). November = worst (–3% avg). SSD (Intel, Samsung): gradual Pyx FMV erosion as flash costs decline. HDD (Seagate, WD): stable with capacity-driven floor. EU E-Waste regulations will incrementally increase used storage supply over 12–18 months.",
        "interpretation": "No EA signals specific to Storage. Price movement is lifecycle and seasonality-driven. Enterprise SSD buyers can afford to wait — flash cost trajectories favour buyers over time.",
        "action": "**Stable — no immediate action required.** Purchase on normal contract cycles. If planning large storage procurements, **October is the historically strongest buying window**. Used storage resellers: factor in E-Waste regulation supply increase over next 12–18 months when modelling resale values.",
        "signals_active": [],
    },
    "NetworkAdapter": {
        "insight": "Network Adapter has the lowest data confidence in this system — only 9 GC_IDs tracked. However, EA2 signals tell a forward-looking story: AI cluster interconnect demand is building and will follow GPU procurement within 1–2 quarters.",
        "finding": "9 GC_IDs total. Mellanox: 1 GC_ID. Broadcom: 3 GC_IDs. Avg monthly volatility: **28%** (highest of all segments — reflects illiquidity, not signal). ANOVA seasonality: NOT significant (p=0.099). EA2: NVIDIA/Mellanox optic partnerships signal next-gen interconnect bottleneck forming for AI cluster builds.",
        "interpretation": "High volatility on thin data = noise, not market direction. The divergence between low data confidence and active EA2 forward signals means the signal is active but price data cannot yet confirm it. Expand GC_ID coverage — do not dismiss the signal.",
        "action": "**WATCH.** High-speed AI interconnect (Mellanox InfiniBand HDR/NDR, 400GbE+): WATCH with FORWARD BUY consideration alongside GPU procurement. **Expand coverage to 30–50 Mellanox/Broadcom SKUs before next sprint.**",
        "signals_active": ["ea2_major_gpu_launch","ea2_datacenter_expansion"],
    },
}

C = {
    "navy":"#1F3864","blue":"#2166AC","red":"#C0392B","green":"#27AE60",
    "orange":"#E08214","purple":"#6C3483","teal":"#148F77","gray":"#5D6D7E",
    "gold":"#F4D03F","lgray":"#D5D8DC",
}
MODEL_COLOR = {"XGBoost":C["blue"],"LightGBM":C["red"],"LSTM":C["green"]}
SEG_COLOR   = {"GPU":C["blue"],"RAM":C["red"],"Processor":C["orange"],"Storage":C["green"],"NetworkAdapter":C["purple"]}

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":8,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.2,"grid.linewidth":0.4,"figure.facecolor":"white",
})

def gpu_tier(row):
    cap = str(row.get("capacity",""))
    m   = _re.search(r"(\d+)\s*GB", cap, _re.IGNORECASE)
    gb  = int(m.group(1)) if m else 0
    t   = str(row.get("type","")).upper()
    if ("GDDR6" in t or "HBM" in t) and gb >= 16: return "AI / Datacenter"
    if gb >= 8 and ("GDDR6" in t or "GDDR5" in t): return "Compute / Workstation"
    return "Legacy / Consumer"

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0); plt.close(fig); return buf

@st.cache_data(ttl=300)
def load_data():
    return _load_all_from_loader()

def load_data_ORIGINAL_REPLACED():
    d = {}
    def _csv(name):
        for p in [f"{DRIVE}/{name}", f"{DRIVE}/sprints/signal_transformation/{name}"]:
            if os.path.exists(p): return pd.read_csv(p, low_memory=False)
        return pd.DataFrame()

    d["momentum"]    = _csv("gc_id_momentum.csv")
    d["category"]    = _csv("category_summary_stats.csv")
    d["metrics"]     = _csv("metrics_all_models.csv")
    d["causal"]      = _csv("causal_analysis_per_gc_id.csv")
    d["test_detail"] = _csv("test_per_gc_id_all_models.csv")

    fv_path = f"{DRIVE}/sprints/signal_transformation/enterprise_feature_vector.csv"
    if not os.path.exists(fv_path): fv_path = f"{DRIVE}/enterprise_feature_vector.csv"
    raw_fv = pd.read_csv(fv_path).iloc[0] if os.path.exists(fv_path) else pd.Series(dtype=float)
    fv = raw_fv.copy()
    for k,v in CURRENT_SIGNALS.items():
        if k not in fv.index or float(fv.get(k,0))==0: fv[k]=v
    d["fv"] = fv

    fc_parts = []
    for seg in SEGMENTS:
        p = f"{DRIVE}/phase3_future_forecast_{seg}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p); df["segment"]=seg; fc_parts.append(df)
    d["forecasts"] = pd.concat(fc_parts,ignore_index=True) if fc_parts else pd.DataFrame()

    # Load augmented files for historical data
    aug_parts = []
    for seg in SEGMENTS:
        p = f"{DRIVE}/augmented_{seg}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p,low_memory=False); df["segment"]=seg; aug_parts.append(df)
    d["augmented"] = pd.concat(aug_parts,ignore_index=True) if aug_parts else pd.DataFrame()

    if not d["momentum"].empty:
        d["momentum"]["gpu_tier"] = d["momentum"].apply(gpu_tier, axis=1)

    return d

def get_best_model(metrics_df, segment):
    """Return the best performing model for a segment based on T+4 MdAPE."""
    if metrics_df.empty or "segment" not in metrics_df.columns: return "XGBoost"
    seg_met = metrics_df[metrics_df["segment"]==segment]
    if "split" in seg_met.columns: seg_met = seg_met[seg_met["split"]=="test"]
    if "horizon" in seg_met.columns: seg_met = seg_met[seg_met["horizon"]=="T+4"]
    if seg_met.empty or "mape_median" not in seg_met.columns: return "XGBoost"
    best = seg_met.loc[seg_met["mape_median"].idxmin()]
    return best.get("model","XGBoost")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PyxTech AI CoE", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main{padding-top:0.5rem}
.signal-on{background:#FDEDEC;border-left:4px solid #C0392B;padding:4px 12px;
           border-radius:3px;margin:2px 0;font-size:.82rem;font-weight:600;color:#922B21}
.signal-off{color:#bbb;font-size:.82rem;padding:2px 12px}
.insight-box{
    background:rgba(33,102,172,0.12);
    border-left:4px solid #2166AC;
    padding:10px 14px;border-radius:4px;margin:8px 0;font-size:.88rem}
.finding-box{
    background:rgba(224,130,20,0.12);
    border-left:4px solid #E08214;
    padding:10px 14px;border-radius:4px;margin:8px 0;font-size:.88rem}
.action-box{
    background:rgba(39,174,96,0.12);
    border-left:4px solid #27AE60;
    padding:10px 14px;border-radius:4px;margin:8px 0;font-size:.88rem}
.uncertainty-box{
    background:rgba(108,52,131,0.12);
    border-left:4px solid #6C3483;
    padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.88rem}
.kpi-note{font-size:.7rem;color:#888;margin-top:-10px}
</style>""", unsafe_allow_html=True)

data = load_data()
fv   = data["fv"]
mom  = data["momentum"]
cat  = data["category"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 PyxTech AI CoE")
    st.caption("Enterprise Hardware Intelligence · March 2026")
    st.divider()
    st.markdown("### 🚨 Market Signal Flags")
    n_active = sum(1 for k in FLAGS.values() if float(fv.get(k,0))>=0.5)
    st.caption(f"**{n_active}/{len(FLAGS)} ACTIVE** — March 2026 sprint")
    for name, key in FLAGS.items():
        if float(fv.get(key,0))>=0.5:
            st.markdown(f'<div class="signal-on">● {name}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-off">○ {name}</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 Pressure Indices")
    for name, key in [("Supply Pressure (EA1)","ea1_supply_pressure_index"),
                      ("AI Demand (EA2)","ea2_ai_demand_index"),
                      ("Policy Risk (EA3)","ea3_policy_risk_index"),
                      ("Combined","combined_market_pressure")]:
        val = float(fv.get(key,0))
        st.markdown(f"**{name}** `{val:.3f}`")
        st.progress(min(val,1.0))
    st.divider()
    st.caption("⚠️ Signal SHAP ~0% after 1 sprint — expected.\nSignals injected via Fix A/B/C.")
    if st.button("🔄 Refresh data"): st.cache_data.clear(); st.rerun()

tab1, tab2, tab3 = st.tabs([
    "📡 What The Market Is Saying",
    "🔭 Forecasts & Pyx FMV",
    "🤖 PyxieAnalyst",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WHAT THE MARKET IS SAYING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("What The Market Is Saying")
    st.caption("EA1 Supply Chain · EA2 AI Technology · EA3 Policy & Trade · March 2026")

    if mom.empty:
        st.warning(f"No gc_id_momentum.csv found at {DRIVE}"); st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    ram_3m   = mom[mom["category"]=="RAM"]["recent_3m_change"].median()
    gpu_df   = mom[mom["category"]=="GPU"]
    ai_gpu   = gpu_df[gpu_df["gpu_tier"]=="AI / Datacenter"]["latest_fmv"]
    gpu_ai_fmv = ai_gpu.mean() if not ai_gpu.empty else gpu_df["latest_fmv"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("RAM 3M Median", f"+{ram_3m:.1f}%", "SHORTAGE ACTIVE")
    c2.metric("Pyx FMV — GPU AI-Tier", f"${gpu_ai_fmv:,.0f}", "H100/A100 secondary mkt")
    c3.metric("GC_IDs Tracked", f"{len(mom):,}")
    c4.metric("Signal Flags Active", f"{n_active}/8")
    c5.metric("Segments Covered", mom["category"].nunique())
    c6.metric("Supply Pressure", f"{float(fv.get('ea1_supply_pressure_index',0.22)):.3f}")
    st.markdown('<p class="kpi-note">GPU Pyx FMV = AI/Datacenter tier only (H100, A100, A6000 ≥16GB GDDR6/HBM). Legacy excluded from benchmark.</p>', unsafe_allow_html=True)
    st.divider()

    # ── SEGMENT TOGGLE ────────────────────────────────────────────────────────
    st.subheader("📡 Segment Deep-Dive — Signal Analysis & Pyx Data")
    seg_labels = {"GPU":"🖥️ GPU","RAM":"💾 RAM","Processor":"⚙️ Processor","Storage":"💿 Storage","NetworkAdapter":"🌐 Network Adapter"}
    seg_sel = st.radio("Select segment to analyse:", list(seg_labels.keys()),
                       format_func=lambda x: seg_labels[x], horizontal=True, key="seg_toggle")

    seg_df   = mom[mom["category"]==seg_sel].copy()
    seg_narr = SEG_NARRATIVE.get(seg_sel, {})

    # Signals status for this segment
    active_sigs = seg_narr.get("signals_active",[])
    inactive_sigs = [k for k in SEG_SIGNALS.get(seg_sel,[]) if k not in active_sigs]

    sig_col, data_col = st.columns([1,2])
    with sig_col:
        st.markdown("**Active EA Signals**")
        if active_sigs:
            for s in active_sigs:
                lbl = s.replace("ea1_","EA1: ").replace("ea2_","EA2: ").replace("ea3_","EA3: ").replace("_"," ").title()
                st.markdown(f'<div class="signal-on">● {lbl}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-off">○ No segment-specific signals active</div>', unsafe_allow_html=True)
        for s in inactive_sigs:
            lbl = s.replace("ea1_","EA1: ").replace("ea2_","EA2: ").replace("ea3_","EA3: ").replace("_"," ").title()
            st.markdown(f'<div class="signal-off">○ {lbl}</div>', unsafe_allow_html=True)

    with data_col:
        if not seg_df.empty:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("GC_IDs", len(seg_df))
            m_col2.metric("Median Pyx FMV", f"${seg_df['latest_fmv'].median():,.0f}")
            m_col3.metric("3M Change", f"{seg_df['recent_3m_change'].median():+.1f}%")
            m_col4.metric("Avg Volatility", f"{seg_df['volatility'].median():.1f}%" if "volatility" in seg_df.columns else "N/A")

    # GPU special: disaggregated view
    if seg_sel == "GPU" and not seg_df.empty:
        st.error("⚠️ **Do NOT use the GPU aggregate median for procurement.** The market is two completely separate markets.")
        g1,g2,g3 = st.columns(3)
        for col, tier, clr in [(g1,"AI / Datacenter","normal"),(g2,"Compute / Workstation","normal"),(g3,"Legacy / Consumer","inverse")]:
            sub = seg_df[seg_df["gpu_tier"]==tier]
            col.metric(f"{tier.split('/')[0].strip()} Pyx FMV",
                       f"${sub['latest_fmv'].mean():,.0f}" if not sub.empty else "N/A",
                       f"3M: {sub['recent_3m_change'].median():+.1f}%" if not sub.empty else "",
                       delta_color=clr)

    # Narrative panels
    if seg_narr:
        st.markdown(f'<div class="insight-box"><b>INSIGHT</b><br>{seg_narr.get("insight","")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="finding-box"><b>FINDING</b><br>{seg_narr.get("finding","")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="action-box"><b>PROCUREMENT ACTION</b><br>{seg_narr.get("action","")}</div>', unsafe_allow_html=True)

    # Charts for segment
    if not seg_df.empty:
        st.subheader(f"📊 {seg_labels[seg_sel]} — Pyx Data Analysis")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # 3M vs 6M momentum scatter
            fig, ax = plt.subplots(figsize=(5.5,3.5))
            col = SEG_COLOR.get(seg_sel, C["blue"])
            m3  = seg_df["recent_3m_change"].clip(-100,300)
            m6  = seg_df["recent_6m_change"].clip(-100,500) if "recent_6m_change" in seg_df.columns else m3
            ax.scatter(m6, m3, color=col, alpha=0.4, s=12, edgecolors="none")
            ax.axhline(0, color=C["gray"], lw=0.7, ls="--")
            ax.axvline(0, color=C["gray"], lw=0.7, ls="--")
            ax.set_xlabel("6-Month Pyx FMV Change (%)")
            ax.set_ylabel("3-Month Pyx FMV Change (%)")
            ax.set_title(f"{seg_sel} — 3M vs 6M Momentum\nTop-right = accelerating (shortage getting worse)")
            ax.text(0.97,0.97,"Top-right = accelerating",transform=ax.transAxes,
                    ha="right",va="top",fontsize=7,color=C["red"],style="italic")
            st.image(fig_to_img(fig), use_container_width=True)

        with chart_col2:
            # FMV distribution, split by tier for GPU
            fig, ax = plt.subplots(figsize=(5.5,3.5))
            if seg_sel == "GPU":
                for tier, col_t in [("AI / Datacenter",C["red"]),("Compute / Workstation",C["blue"]),("Legacy / Consumer",C["gray"])]:
                    sub = seg_df[seg_df["gpu_tier"]==tier]["recent_3m_change"].dropna().clip(-80,80)
                    if not sub.empty:
                        ax.hist(sub, bins=20, alpha=0.6, color=col_t,
                                label=f"{tier.split('/')[0].strip()} (med {sub.median():+.0f}%)",
                                density=True, edgecolor="white", linewidth=0.3)
                ax.set_title("GPU 3M Momentum by Tier\n⚠️ Aggregate misleads — always use tier view")
            else:
                m3_plot = seg_df["recent_3m_change"].dropna().clip(-100,200)
                ax.hist(m3_plot, bins=35, color=SEG_COLOR.get(seg_sel,C["blue"]), alpha=0.75,
                        density=True, edgecolor="white", linewidth=0.3)
                ax.axvline(m3_plot.median(), color=C["navy"], lw=1.8, ls="--",
                           label=f"Median {m3_plot.median():+.1f}%")
                ax.axvline(m3_plot.mean(), color=C["orange"], lw=1.5, ls=":",
                           label=f"Mean {m3_plot.mean():+.1f}%")
                ax.set_title(f"{seg_sel} — 3M Pyx FMV Change Distribution")
            ax.axvline(0, color=C["gray"], lw=0.8)
            ax.set_xlabel("3-Month Pyx FMV Change (%)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            st.image(fig_to_img(fig), use_container_width=True)

        # Top/bottom movers
        st.markdown(f"**Top & Bottom Movers — {seg_labels[seg_sel]}**")
        label_col = "model_name" if "model_name" in seg_df.columns else "gc_id"
        movers = seg_df.nlargest(5,"recent_3m_change")[[label_col,"manufacturer","recent_3m_change","latest_fmv"]].copy()
        losers = seg_df.nsmallest(5,"recent_3m_change")[[label_col,"manufacturer","recent_3m_change","latest_fmv"]].copy()
        movers["Direction"] = "▲ Appreciating"
        losers["Direction"] = "▼ Declining"
        combined = pd.concat([movers,losers]).rename(columns={
            label_col:"Model","manufacturer":"Manufacturer","recent_3m_change":"3M Change %","latest_fmv":"Pyx FMV $"})
        st.dataframe(combined.style.format({"3M Change %":"{:+.1f}%","Pyx FMV $":"${:,.0f}"}),
                     use_container_width=True, hide_index=True)

    st.divider()
    # Cross-segment momentum
    st.subheader("📊 Cross-Segment Pyx FMV Momentum — Signal Overlay")
    segs_k = ["GPU","RAM","Processor","Storage","Network_Adapter"]
    segs_l = ["GPU\n(aggregate⚠️)","RAM","Processor","Storage","Network\nAdapter"]
    mom_3m_all = mom.groupby("category")["recent_3m_change"].median()
    vals = [mom_3m_all.get(s,0) for s in segs_k]
    colors = [C["red"] if k=="RAM" else (C["orange"] if v>0 else C["blue"]) for k,v in zip(segs_k,vals)]

    fig, ax = plt.subplots(figsize=(9,3.5))
    bars = ax.barh(segs_l, vals, color=colors, alpha=0.85, height=0.55)
    ax.axvline(0, color=C["gray"], lw=0.8)
    for bar,val in zip(bars,vals):
        off = 1 if val>=0 else -1
        ax.text(val+off, bar.get_y()+bar.get_height()/2, f"{val:+.1f}%",
                va="center", ha="left" if val>=0 else "right",
                fontsize=8, fontweight="bold")
    ax.set_xlabel("Median 3-Month Pyx FMV Change (%)")
    ax.set_title("⚠️ GPU aggregate is misleading — always use disaggregated tier Pyx FMV for GPU procurement",
                 fontsize=7, style="italic")
    st.image(fig_to_img(fig), use_container_width=True)

    # Procurement decision matrix
    st.subheader("🎯 Full Procurement Decision Matrix")
    st.caption("⚠️ **Guidance only.** The directional signals below are data-driven observations from PyxTech's pipeline. They are not financial or procurement advice. All purchasing decisions remain the sole responsibility of your organisation. Past price trends do not guarantee future performance.")
    st.dataframe(pd.DataFrame([
        {"Segment":"GPU AI-tier (H100/A100/A6000)","Pyx FMV":"$17,087+ secondary","Action":"▲ Elevated Priority — Review Procurement","Direction":"▲▲ +10–25%","Confidence":"HIGH","Rationale":"Rotation not relief. Floor established."},
        {"Segment":"RAM DDR5/HBM","Pyx FMV":"$48 median (raw)","Action":"▲▲ Urgent — Review Procurement Immediately","Direction":"▲▲ +15–30%","Confidence":"HIGH","Rationale":"+120% 6M. No spot market. Accelerating."},
        {"Segment":"RAM DDR4","Pyx FMV":"$48 median (raw)","Action":"▲ Elevated Priority — Review Procurement","Direction":"▲▲ +15–30%","Confidence":"HIGH","Rationale":"+56% 6M. EA1+price converged."},
        {"Segment":"GPU Legacy (K/Tesla/Quadro)","Pyx FMV":"$1–$50","Action":"▼ Low Priority — Reassess Valuation","Direction":"▼▼ –30–90%","Confidence":"HIGH","Rationale":"Structural freefall. Approaching scrap."},
        {"Segment":"Processor Scalable 3rd/4th gen","Pyx FMV":"$50–$200","Action":"→ Monitor — Review at Next PO Cycle","Direction":"→ Flat ±3%","Confidence":"MED","Rationale":"Stable. Normal cycles."},
        {"Segment":"Processor Legacy (Xeon E5/Pentium)","Pyx FMV":"$1–$3","Action":"▼ Low Priority — Reassess Valuation","Direction":"▼▼ –80–90%","Confidence":"HIGH","Rationale":"EOL. $1–3/unit. Never price above scrap."},
        {"Segment":"Storage","Pyx FMV":"$50–$150","Action":"→ Stable — No Immediate Action Required","Direction":"▲ +2–5%","Confidence":"LOW","Rationale":"No shortage. Oct = best seasonal window."},
        {"Segment":"Network Adapter","Pyx FMV":"$14–$248","Action":"→ Monitor — Review at Next PO Cycle","Direction":"→ ±5%","Confidence":"LOW","Rationale":"9 GC_IDs only. Expand coverage first."},
    ]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECASTS & PYX FMV
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("4-Month Pyx FMV Forecasts")
    st.caption("XGBoost · LightGBM · LSTM | Signal-augmented (Fix A+B+C) | March–June 2026")

    # ── Plain-English explainer ────────────────────────────────────────────────
    with st.expander("📖 How to read these forecasts — plain English guide", expanded=False):
        st.markdown("""
**What is Pyx FMV?**
Pyx FMV (Fair Market Value) is PyxTech's estimate of what a specific piece of enterprise hardware would sell for in the secondary market today, under normal market conditions. It is based on 50 months of historical transaction data per GC_ID.

**What is Uncertainty %?**
Uncertainty tells you how wide the range of possible future prices is, expressed as a percentage of the forecast value.
- **Low uncertainty (0–20%)** = the model is confident. Example: 10% uncertainty on a $100 Pyx FMV means the actual price is likely between $90–$110.
- **Medium uncertainty (20–50%)** = the model sees meaningful price variability. Treat the forecast as directional guidance, not a precise number.
- **High uncertainty (>50%)** = the model is uncertain. This often happens for thin-data segments (few GC_IDs), highly volatile SKUs, or AI-tier GPU where prices can move sharply.

**What are the 5% and 95% Confidence Intervals?**
These are the lower and upper bounds of the forecast range.
- The **5% CI** is the floor — there is only a 5% chance the actual price will be *below* this.
- The **95% CI** is the ceiling — there is only a 5% chance the actual price will be *above* this.
- The wider this band, the less certain the forecast.

**Which model should I trust?**
The system auto-selects the best-performing model for each segment based on test-set accuracy (MdAPE = Median Absolute Percentage Error). Lower MdAPE = better. We show all three models so you can see where they agree (higher confidence) and where they diverge (higher uncertainty).

**Important limitation:** Signal SHAP attribution is currently ~0% because only 1 sprint of EA market signals has been collected. The 3–6 additional sprints will allow signals (export controls, DRAM shortage, GPU launch) to materially influence forecast outputs. Current forecasts are driven by price history patterns.
        """)

    st.divider()

    # ── Section A: GC_ID Search ────────────────────────────────────────────────
    st.subheader("🔍 Section A: Find Pyx FMV Forecast for a Specific SKU")
    st.caption("Search by GC_ID or filter by component attributes to get a specific SKU forecast.")

    fc_df = data["forecasts"]
    if fc_df.empty:
        st.warning("No phase3_future_forecast_*.csv files found."); st.stop()

    search_mode = st.radio("Search method:", ["Filter by attributes","Enter GC_ID directly"], horizontal=True)

    if search_mode == "Enter GC_ID directly":
        gc_input = st.text_input("Enter GC_ID (exact match):", placeholder="e.g. 12345")
        target_gc_ids = [gc_input.strip()] if gc_input.strip() else []
    else:
        if mom.empty:
            st.warning("No momentum data available for filtering."); st.stop()
        f1,f2,f3,f4 = st.columns(4)
        seg_filt = f1.selectbox("Segment", ["All"]+SEGMENTS, key="seg_filt")
        mfr_opts = ["All"]+sorted(mom["manufacturer"].dropna().unique().tolist()) if "manufacturer" in mom.columns else ["All"]
        mfr_filt = f2.selectbox("Manufacturer", mfr_opts, key="mfr_filt")

        filtered_mom = mom.copy()
        if seg_filt != "All": filtered_mom = filtered_mom[filtered_mom["category"]==seg_filt]
        if mfr_filt != "All": filtered_mom = filtered_mom[filtered_mom["manufacturer"]==mfr_filt]

        f5,f6 = st.columns(2)
        type_opts = ["All"]+sorted(filtered_mom["type"].dropna().unique().tolist()) if "type" in filtered_mom.columns else ["All"]
        type_filt = f5.selectbox("Type", type_opts, key="type_filt")
        if type_filt != "All": filtered_mom = filtered_mom[filtered_mom["type"]==type_filt]

        if seg_filt=="GPU" and "gpu_tier" in filtered_mom.columns:
            tier_opts = ["All"]+sorted(filtered_mom["gpu_tier"].dropna().unique().tolist())
            tier_filt = f6.selectbox("GPU Tier", tier_opts, key="tier_filt")
            if tier_filt != "All": filtered_mom = filtered_mom[filtered_mom["gpu_tier"]==tier_filt]

        model_opts = ["All"]+sorted(filtered_mom["model_name"].dropna().unique().tolist()) if "model_name" in filtered_mom.columns else ["All"]
        model_filt = st.selectbox("Model Name (optional)", model_opts[:100], key="model_name_filt")
        if model_filt != "All": filtered_mom = filtered_mom[filtered_mom["model_name"]==model_filt]

        st.caption(f"**{len(filtered_mom)} GC_IDs match your filters.** Select up to 5 to compare.")
        if not filtered_mom.empty:
            label_col = "model_name" if "model_name" in filtered_mom.columns else "gc_id"
            filtered_mom["display"] = filtered_mom.apply(
                lambda r: f"{r.get('gc_id','')} | {r.get('model_name','?')} {r.get('model_number','')} | {r.get('manufacturer','')} | Pyx FMV ${r.get('latest_fmv',0):,.0f} | 3M {r.get('recent_3m_change',0):+.1f}%",
                axis=1)
            selected_display = st.multiselect("Select GC_IDs:", filtered_mom["display"].tolist()[:200],
                                               max_selections=5, key="gc_multi_sel")
            target_gc_ids = [d.split(" | ")[0].strip() for d in selected_display]
        else:
            target_gc_ids = []

    # Show forecast for selected GC_IDs
    if target_gc_ids:
        for gc_id in target_gc_ids:
            gc_id = str(gc_id)
            gc_fc = fc_df[fc_df["gc_id"].astype(str)==gc_id] if "gc_id" in fc_df.columns else pd.DataFrame()
            gc_meta = mom[mom["gc_id"].astype(str)==gc_id].iloc[0] if not mom.empty and gc_id in mom["gc_id"].astype(str).values else pd.Series()

            st.markdown(f"---")
            st.markdown(f"#### GC_ID {gc_id}" + (f" — {gc_meta.get('model_name','')} {gc_meta.get('model_number','')} ({gc_meta.get('manufacturer','')})" if not gc_meta.empty else ""))

            if not gc_meta.empty:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Current Pyx FMV", f"${gc_meta.get('latest_fmv',0):,.0f}")
                m2.metric("3M Change", f"{gc_meta.get('recent_3m_change',0):+.1f}%")
                m3.metric("6M Change", f"{gc_meta.get('recent_6m_change',0):+.1f}%" if "recent_6m_change" in gc_meta.index else "N/A")
                m4.metric("Volatility", f"{gc_meta.get('volatility',0):.1f}%" if "volatility" in gc_meta.index else "N/A")
                if seg_filt=="GPU" and "gpu_tier" in gc_meta.index:
                    st.caption(f"GPU Tier: **{gc_meta.get('gpu_tier','?')}** | Type: {gc_meta.get('type','?')} | Region: {gc_meta.get('region','?')}")

            if not gc_fc.empty:
                # Auto-select best model
                best_model = get_best_model(data["metrics"], gc_meta.get("category","GPU") if not gc_meta.empty else "GPU")
                best_fc = gc_fc[gc_fc["model"]==best_model] if "model" in gc_fc.columns else gc_fc

                fig, ax = plt.subplots(figsize=(8,3.5))
                current_fmv = gc_meta.get("latest_fmv",0) if not gc_meta.empty else 0
                if current_fmv: ax.axhline(current_fmv, color=C["gray"], lw=1.0, ls="--", alpha=0.5, label=f"Current Pyx FMV ${current_fmv:,.0f}")

                h_order = {"T+1":1,"T+2":2,"T+3":3,"T+4":4}
                for model in MODELS:
                    m_df = gc_fc[gc_fc["model"]==model] if "model" in gc_fc.columns else gc_fc
                    if m_df.empty: continue
                    pts = m_df.sort_values("horizon", key=lambda x: x.map(h_order))
                    lw  = 2.5 if model==best_model else 1.2
                    ls  = "-" if model==best_model else "--"
                    lbl = f"{'★ ' if model==best_model else ''}{model}"
                    ax.plot(pts["horizon"], pts["point_forecast"], color=MODEL_COLOR.get(model,C["gray"]),
                            lw=lw, ls=ls, marker="o", markersize=5 if model==best_model else 3, label=lbl)
                    if "ci_lower_95" in m_df.columns and model==best_model:
                        ax.fill_between(pts["horizon"], pts["ci_lower_95"], pts["ci_upper_95"],
                                        alpha=0.15, color=MODEL_COLOR.get(model,C["gray"]))

                ax.set_ylabel("Pyx FMV ($)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
                ax.set_title(f"GC_ID {gc_id} — 4-Month Pyx FMV Forecast\n★ = best model for this segment | Shaded area = 90% CI range")
                ax.legend(fontsize=7)
                st.image(fig_to_img(fig), use_container_width=True)

                # Forecast table
                tbl_rows = []
                for h in HORIZONS:
                    for model in MODELS:
                        m_df = gc_fc[(gc_fc["horizon"]==h)&(gc_fc["model"]==model)] if "model" in gc_fc.columns else gc_fc[gc_fc["horizon"]==h]
                        if m_df.empty: continue
                        row = m_df.iloc[0]
                        unc = float(row.get("uncertainty_pct",0)) if "uncertainty_pct" in row.index else None
                        lo  = float(row.get("ci_lower_95",0)) if "ci_lower_95" in row.index else None
                        hi  = float(row.get("ci_upper_95",0)) if "ci_upper_95" in row.index else None
                        tbl_rows.append({
                            "Horizon":h, "Model":("★ " if model==best_model else "")+model,
                            "Pyx FMV Forecast":float(row.get("point_forecast",0)),
                            "5% CI (floor)":lo, "95% CI (ceiling)":hi,
                            "Uncertainty %":unc,
                        })
                if tbl_rows:
                    tbl = pd.DataFrame(tbl_rows)
                    st.dataframe(tbl.style.format({
                        "Pyx FMV Forecast":"${:,.0f}","5% CI (floor)":"${:,.0f}",
                        "95% CI (ceiling)":"${:,.0f}","Uncertainty %":"{:.1f}%"
                    }), use_container_width=True, hide_index=True)

                # Uncertainty interpretation
                if tbl_rows:
                    best_t4 = [r for r in tbl_rows if r["Horizon"]=="T+4" and best_model in r["Model"]]
                    if best_t4 and best_t4[0].get("Uncertainty %"):
                        unc_val = best_t4[0]["Uncertainty %"]
                        unc_level = "LOW" if unc_val < 20 else "MEDIUM" if unc_val < 50 else "HIGH"
                        unc_msg = {
                            "LOW": f"**Uncertainty: LOW ({unc_val:.1f}%)** — The model is confident in this forecast. The actual Pyx FMV at T+4 is likely to be within ±{unc_val/2:.0f}% of the forecast value.",
                            "MEDIUM": f"**Uncertainty: MEDIUM ({unc_val:.1f}%)** — There is meaningful price variability expected. Use the 5%–95% CI range as your planning envelope, not just the single forecast number.",
                            "HIGH": f"**Uncertainty: HIGH ({unc_val:.1f}%)** — The model sees significant price uncertainty ahead. This could be due to thin historical data, high volatility, or active market disruption (e.g. AI-tier GPU). Treat the forecast as directional — plan for the range, not the point estimate.",
                        }[unc_level]
                        st.markdown(f'<div class="uncertainty-box">{unc_msg}</div>', unsafe_allow_html=True)
            else:
                st.info(f"No forecast data found for GC_ID {gc_id}. This GC_ID may not have been included in the forecasting run.")

    st.divider()

    # ── Section B: Group Analysis ─────────────────────────────────────────────
    st.subheader("📊 Section B: Group Analysis — Pyx FMV by Manufacturer / Tier / Region")
    st.caption("Higher-level view: how does the forecast look when grouped? Auto-selects best performing model per segment.")

    seg_grp = st.selectbox("Select segment for group analysis:", SEGMENTS, key="seg_grp",
                           format_func=lambda x: x.replace("NetworkAdapter","Network Adapter"))
    grp_by  = st.radio("Group by:", ["Manufacturer","GPU Tier","Type","Region"], horizontal=True, key="grp_by")

    grp_fc  = fc_df[fc_df["segment"]==seg_grp].copy()
    grp_mom = mom[mom["category"]==seg_grp].copy()
    best_m  = get_best_model(data["metrics"], seg_grp)

    if not grp_fc.empty and not grp_mom.empty:
        # Merge forecast with metadata
        if "gc_id" in grp_fc.columns and "gc_id" in grp_mom.columns:
            meta_cols = ["gc_id","manufacturer","type","region","gpu_tier","model_name","latest_fmv","recent_3m_change"]
            meta_cols = [c for c in meta_cols if c in grp_mom.columns]
            grp_fc = grp_fc.merge(grp_mom[meta_cols], on="gc_id", how="left")

        grp_col_map = {
            "Manufacturer":"manufacturer","GPU Tier":"gpu_tier",
            "Type":"type","Region":"region"
        }
        grp_col = grp_col_map.get(grp_by,"manufacturer")

        if grp_col in grp_fc.columns and "model" in grp_fc.columns:
            best_grp_fc = grp_fc[grp_fc["model"]==best_m]
            t4_grp = best_grp_fc[best_grp_fc["horizon"]=="T+4"].dropna(subset=[grp_col,"point_forecast"])

            if not t4_grp.empty:
                grp_summary = t4_grp.groupby(grp_col).agg(
                    median_forecast=("point_forecast","median"),
                    mean_forecast=("point_forecast","mean"),
                    n_gc_ids=("gc_id","nunique") if "gc_id" in t4_grp.columns else ("point_forecast","count"),
                    avg_uncertainty=("uncertainty_pct","mean") if "uncertainty_pct" in t4_grp.columns else ("point_forecast","std"),
                ).reset_index().sort_values("median_forecast", ascending=True)

                # Also get current Pyx FMV for comparison
                if grp_col in grp_mom.columns:
                    current = grp_mom.groupby(grp_col)["latest_fmv"].median().reset_index().rename(columns={"latest_fmv":"current_pyx_fmv"})
                    grp_summary = grp_summary.merge(current, on=grp_col, how="left")
                    grp_summary["pct_change_T4"] = (grp_summary["median_forecast"] - grp_summary["current_pyx_fmv"]) / grp_summary["current_pyx_fmv"].abs() * 100

                g_col1, g_col2 = st.columns([1,1])
                with g_col1:
                    fig, ax = plt.subplots(figsize=(6,max(3, len(grp_summary)*0.5+1)))
                    colors_grp = [C["red"] if v>0 else C["blue"]
                                  for v in grp_summary.get("pct_change_T4", pd.Series([0]*len(grp_summary)))]
                    ax.barh(grp_summary[grp_col].astype(str), grp_summary["median_forecast"],
                            color=C["blue"], alpha=0.75, height=0.6)
                    if "current_pyx_fmv" in grp_summary.columns:
                        ax.barh(grp_summary[grp_col].astype(str), grp_summary["current_pyx_fmv"],
                                color=C["gray"], alpha=0.4, height=0.6, label="Current Pyx FMV")
                    ax.set_xlabel("Pyx FMV ($)")
                    ax.set_title(f"{seg_grp} — T+4 Forecast vs Current\nby {grp_by} | Model: {best_m} ★")
                    ax.legend(fontsize=7)
                    st.image(fig_to_img(fig), use_container_width=True)

                with g_col2:
                    display_cols = [grp_col,"n_gc_ids","current_pyx_fmv","median_forecast"]
                    if "pct_change_T4" in grp_summary.columns: display_cols.append("pct_change_T4")
                    if "avg_uncertainty" in grp_summary.columns: display_cols.append("avg_uncertainty")
                    display_cols = [c for c in display_cols if c in grp_summary.columns]
                    rename_map = {grp_col:grp_by.replace("GPU Tier","Tier"),"n_gc_ids":"GC_IDs",
                                  "current_pyx_fmv":"Current Pyx FMV","median_forecast":f"T+4 Forecast ({best_m}★)",
                                  "pct_change_T4":"Forecast Change %","avg_uncertainty":"Avg Uncertainty %"}
                    fmt_map = {"Current Pyx FMV":"${:,.0f}",f"T+4 Forecast ({best_m}★)":"${:,.0f}",
                               "Forecast Change %":"{:+.1f}%","Avg Uncertainty %":"{:.1f}%"}
                    st.dataframe(grp_summary[display_cols].rename(columns=rename_map)
                                 .style.format({k:v for k,v in fmt_map.items() if k in grp_summary.rename(columns=rename_map).columns}),
                                 use_container_width=True, hide_index=True)

                # Signal alignment
                st.markdown(f'<div class="insight-box"><b>Signal Alignment Check — {seg_grp} {grp_by} Group</b><br>{SEG_NARRATIVE.get(seg_grp,{}).get("interpretation","Signal analysis available per-segment in Tab 1.")}</div>', unsafe_allow_html=True)

                # Uncertainty note
                if "avg_uncertainty" in grp_summary.columns:
                    avg_unc = grp_summary["avg_uncertainty"].mean()
                    st.markdown(f'<div class="uncertainty-box"><b>Average Group Uncertainty: {avg_unc:.1f}%</b><br>{"LOW — forecasts are reliable at the group level." if avg_unc<20 else "MEDIUM — use the CI range for planning, not just the median." if avg_unc<50 else "HIGH — significant variability within this group. Individual GC_ID forecasts (Section A) may give clearer signal."}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PYXIEANALYST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🤖 PyxieAnalyst")
    st.caption("Powered by GPT-4o · Your AI-driven hardware market intelligence analyst")

    @st.cache_data(ttl=300)
    def build_rich_context():
        d = load_data()
        fv_l=d["fv"]; cat_l=d["category"]; mom_l=d["momentum"]; fc_l=d["forecasts"]; met_l=d["metrics"]
        parts = []

        parts.append("=== ACTIVE MARKET SIGNALS — March 2026 (ground truth) ===")
        for k,v in CURRENT_SIGNALS.items():
            parts.append(f"  {k}: {v}")

        # GPU disaggregated
        if not mom_l.empty and "gpu_tier" in mom_l.columns:
            parts.append("\n=== GPU Pyx FMV BY TIER (always use these, never aggregate) ===")
            gpu_m = mom_l[mom_l["category"]=="GPU"]
            parts.append(f"  GPU aggregate: MISLEADING — two separate markets, use tiers below")
            for tier in ["AI / Datacenter","Compute / Workstation","Legacy / Consumer"]:
                sub = gpu_m[gpu_m["gpu_tier"]==tier]
                if not sub.empty:
                    parts.append(f"  {tier}: n={len(sub)} GC_IDs | Pyx FMV mean=${sub['latest_fmv'].mean():,.0f} | median=${sub['latest_fmv'].median():,.0f} | 3M {sub['recent_3m_change'].median():+.1f}%")
            parts.append("  AI-tier secondary market EA2: avg $17,087, peak $51,826 (H100/A100/A6000)")

        # All category stats
        if not cat_l.empty:
            parts.append("\n=== CATEGORY Pyx FMV STATISTICS ===")
            for _,row in cat_l.iterrows():
                parts.append(f"  {row.get('category')}: {row.get('n_gc_ids')} GC_IDs | Pyx FMV ${row.get('median_fmv',0):,.0f} | 3M {row.get('median_3m_change',0):+.1f}% | 6M {row.get('median_6m_change',0):+.1f}% | Convergence: {row.get('signal_price_convergence','?')}")

        # RAM detail
        if not mom_l.empty:
            ram_m = mom_l[mom_l["category"]=="RAM"]["recent_3m_change"]
            parts.append(f"\n=== RAM SHORTAGE DETAIL ===")
            parts.append(f"  3M median +{ram_m.median():.1f}% | mean +{ram_m.mean():.1f}% | max +{ram_m.max():.1f}%")
            parts.append("  DDR4 6M: +56% | DDR5 6M: +120% | HBM3E: no spot market")
            parts.append("  Used enterprise RAM: $5,910/unit average (secondary premium)")

        # Top 5 movers per segment
        for seg in SEGMENTS:
            if mom_l.empty: break
            seg_m = mom_l[mom_l["category"]==seg]
            if seg_m.empty: continue
            parts.append(f"\n=== {seg} TOP 5 MOVERS (3M) ===")
            label_c = "model_name" if "model_name" in seg_m.columns else "gc_id"
            for _,row in seg_m.nlargest(5,"recent_3m_change").iterrows():
                parts.append(f"  {row.get(label_c,'?')} | GC_ID {row.get('gc_id','?')} | Pyx FMV ${row.get('latest_fmv',0):,.0f} | 3M {row.get('recent_3m_change',0):+.1f}%")

        # Forecast T+4 per segment, best model
        if not fc_l.empty and "horizon" in fc_l.columns and "model" in fc_l.columns and not met_l.empty:
            parts.append("\n=== 4-MONTH Pyx FMV FORECASTS (T+4, BEST MODEL PER SEGMENT) ===")
            for seg in SEGMENTS:
                best = get_best_model(met_l, seg)
                t4 = fc_l[(fc_l["segment"]==seg)&(fc_l["horizon"]=="T+4")&(fc_l["model"]==best)]
                if t4.empty: continue
                pt  = t4["point_forecast"].median()
                lo  = t4["ci_lower_95"].quantile(0.1) if "ci_lower_95" in t4.columns else None
                hi  = t4["ci_upper_95"].quantile(0.9) if "ci_upper_95" in t4.columns else None
                unc = t4["uncertainty_pct"].median() if "uncertainty_pct" in t4.columns else None
                parts.append(f"  {seg} [{best}★ best model] T+4 median Pyx FMV: ${pt:,.0f}" +
                             (f" | 90% CI: ${lo:,.0f}–${hi:,.0f}" if lo and hi else "") +
                             (f" | Uncertainty: {unc:.1f}%" if unc else ""))

        # Model accuracy
        if not met_l.empty:
            parts.append("\n=== MODEL ACCURACY (TEST SET T+4) ===")
            t4m = met_l[met_l["horizon"]=="T+4"] if "horizon" in met_l.columns else met_l
            if "split" in t4m.columns: t4m = t4m[t4m["split"]=="test"]
            for _,row in t4m.iterrows():
                mape = row.get("mape_median","?")
                parts.append(f"  {row.get('segment')} {row.get('model')}: MdAPE {mape:.1f}%" if isinstance(mape,float) else "")

        return "\n".join(parts)

    SYSTEM = """You are PyxieAnalyst — the AI market intelligence analyst for PyxTech's enterprise hardware valuation platform.

IDENTITY AND ROLE:
You analyse enterprise hardware pricing data from 2,464 GC_IDs across GPU, RAM, Processor, Storage, and Network Adapter segments. You are grounded in real data from the PyxTech forecasting pipeline (XGBoost/LightGBM/LSTM models) and market signals from three intelligence agents (EA1 supply chain, EA2 AI technology, EA3 policy & trade).

MANDATORY RULES:
1. Always say "Pyx FMV" not just "FMV" or "price" when quoting fair market values
2. ALWAYS disaggregate GPU into AI/Datacenter, Compute/Workstation, and Legacy tiers — never quote aggregate GPU Pyx FMV for procurement
3. When asked about forecasts, cite specific GC_IDs or groups with their actual T+4 Pyx FMV and uncertainty %, not just vague direction
4. Explain uncertainty and confidence intervals in plain English — the audience is procurement professionals, not data scientists
5. Signal SHAP attribution is ~0% after 1 sprint — acknowledge this but explain signals ARE active via injection (Fix A/B/C)
6. Always acknowledge when data is thin (e.g. Network Adapter = 9 GC_IDs) and flag low confidence explicitly
7. Processor: never quote the –5.4% aggregate. Always split Scalable vs Legacy with separate Pyx FMV values

REASONING STYLE:
- Lead with a direct, specific answer using exact numbers from the data context
- Then explain the reasoning with 2–3 key supporting data points
- End with a clear procurement recommendation where relevant
- If the user's question is ambiguous, ask ONE clarifying question — then answer directly
- After your second response, always invite the user to: (1) dig deeper on a specific GC_ID, (2) explore another segment, or (3) ask about the procurement strategy
- Be warm, expert, and conversational — not a data dump

CONVERSATION MANAGEMENT:
- Maximum 2 clarification rounds before giving a best-effort answer
- After answering, always offer 2–3 natural next conversation directions
- If user seems satisfied, proactively suggest related analysis they haven't thought to ask about

Current pipeline data context:
{context}
"""

    # Suggested questions
    st.markdown("**💬 Start here — suggested questions:**")
    suggested = [
        "What is the Pyx FMV for AI-tier GPU and where is it heading?",
        "Give me the top 3 most urgent procurement actions right now.",
        "Why is RAM showing such extreme price movement? What does the data say?",
        "What is the T+4 Pyx FMV forecast for RAM and how confident are we?",
        "Explain what uncertainty means for the GPU AI-tier forecast.",
        "Which GC_IDs have the highest forecast uncertainty and why?",
        "What is Blackwell Ultra doing to secondary GPU market pricing?",
        "Compare Xeon Scalable vs Xeon E5 — what should procurement do?",
    ]
    s_cols = st.columns(4)
    for j, prompt in enumerate(suggested):
        if s_cols[j%4].button(prompt[:42]+"...", key=f"s_{j}"):
            st.session_state.setdefault("messages_v3",[])
            st.session_state["messages_v3"].append({"role":"user","content":prompt})

    st.divider()

    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY",""))
    if not api_key:
        st.warning("Set `OPENAI_API_KEY` before launching PyxieAnalyst.")
    else:
        if "messages_v3" not in st.session_state: st.session_state["messages_v3"]=[]

        for msg in st.session_state["messages_v3"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask PyxieAnalyst about Pyx FMV, signals, specific SKUs, or procurement strategy...")

        if user_input:
            st.session_state["messages_v3"].append({"role":"user","content":user_input})
            with st.chat_message("user"): st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("PyxieAnalyst is analysing..."):
                    try:
                        from openai import OpenAI
                        client  = OpenAI(api_key=api_key)
                        context = build_rich_context()
                        system  = SYSTEM.format(context=context)

                        msgs = [{"role":"system","content":system}]
                        for m in st.session_state["messages_v3"][:-1]:
                            if m["role"] in ("user","assistant"): msgs.append(m)
                        msgs.append({"role":"user","content":user_input})

                        # Count clarification rounds
                        q_count = sum(1 for m in st.session_state["messages_v3"] if m["role"]=="assistant" and "?" in m.get("content",""))
                        if q_count >= 2:
                            msgs.append({"role":"system","content":"You have already asked 2 clarifying questions. Do NOT ask more — give your best answer now with the available information."})

                        stream = client.chat.completions.create(
                            model="gpt-4o", messages=msgs, max_tokens=1800, stream=True)
                        resp = ""
                        ph   = st.empty()
                        for chunk in stream:
                            resp += chunk.choices[0].delta.content or ""
                            ph.markdown(resp+"▌")
                        ph.markdown(resp)
                        st.session_state["messages_v3"].append({"role":"assistant","content":resp})

                    except Exception as e:
                        st.error(f"PyxieAnalyst error: {e}")

        if st.session_state.get("messages_v3"):
            col_a, col_b = st.columns([1,4])
            if col_a.button("🗑️ Clear conversation"):
                st.session_state["messages_v3"]=[]; st.rerun()
            col_b.caption("💡 Pro tip: Ask PyxieAnalyst to compare specific GC_IDs, explain a forecast, or give a procurement recommendation for your next PO cycle.")
