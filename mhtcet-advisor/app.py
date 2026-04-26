"""
app.py — MHT-CET College Preference Advisor
Main Streamlit application entry point.
"""

import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import numpy as np

from src.data_loader import (
    load_all_cutoffs, load_seat_matrix, load_config,
    get_available_branches, get_available_colleges,
)
from src.probability_engine import (
    generate_all_predictions, generate_preference_list,
    float_freeze_advice, classify
)
from src.export import generate_pdf

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MHT-CET Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F8FAFC; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2D5282 100%);
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label { color: #CBD5E0 !important; font-size: 0.82rem; }

    /* Cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-left: 4px solid #3B82F6;
        margin-bottom: 8px;
    }
    .stat-card h3 { font-size: 1.6rem; margin: 0; color: #1E3A5F; }
    .stat-card p  { font-size: 0.78rem; color: #64748B; margin: 0; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    .badge-reach   { background: #EF4444; }
    .badge-dream   { background: #F97316; }
    .badge-target  { background: #3B82F6; }
    .badge-safe    { background: #22C55E; }
    .badge-assured { background: #6B7280; }

    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E3A5F;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }

    /* Preference row */
    .pref-row {
        background: white;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-left: 3px solid #E2E8F0;
    }

    /* Advice boxes */
    .advice-freeze { background: #ECFDF5; border: 1px solid #22C55E; border-radius:8px; padding:14px; }
    .advice-float  { background: #EFF6FF; border: 1px solid #3B82F6; border-radius:8px; padding:14px; }
    .advice-slide  { background: #FFF7ED; border: 1px solid #F97316; border-radius:8px; padding:14px; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #E2E8F0;
        border-radius: 6px 6px 0 0;
        padding: 6px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1E3A5F;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def get_data():
    cutoffs = load_all_cutoffs()
    seat_matrix = load_seat_matrix()
    config = load_config()
    return cutoffs, seat_matrix, config


with st.spinner("Loading historical data…"):
    cutoff_df, seat_matrix_df, config = get_data()


# ── Build university → college mapping ────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_university_map(_cutoff_df, _config):
    """Map each college name to its likely home university using keyword matching."""
    if _cutoff_df.empty:
        return {}
    univ_keywords = _config.get("university_college_keywords", {})
    colleges = _cutoff_df['college_name'].unique()
    mapping = {}
    for college in colleges:
        college_lower = college.lower()
        matched = None
        for univ, keywords in univ_keywords.items():
            if any(kw.lower() in college_lower for kw in keywords):
                matched = univ
                break
        mapping[college] = matched or ""
    return mapping


university_map = build_university_map(cutoff_df, config)


# ── Sidebar — Student Profile ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 MHT-CET Advisor")
    st.markdown("---")

    st.markdown("### 📋 Your Profile")

    percentile = st.number_input(
        "Your Percentile", min_value=0.0, max_value=100.0,
        value=85.0, step=0.01, format="%.2f"
    )

    categories_map = config.get("categories", {})
    category = st.selectbox(
        "Category",
        options=list(categories_map.keys()),
        format_func=lambda k: categories_map[k]
    )

    gender = st.radio("Gender", ["male", "female"], format_func=str.capitalize, horizontal=True)

    # District → University
    district_univ_map = config.get("district_university_map", {})
    districts = sorted(district_univ_map.keys())
    district = st.selectbox("Your Home District", districts)
    home_university = district_univ_map.get(district, "")
    if home_university:
        st.caption(f"🏛️ {home_university}")

    special_quotas_map = config.get("special_quotas", {})
    special_quotas = st.multiselect(
        "Special Quota (if eligible)",
        options=list(special_quotas_map.keys()),
        format_func=lambda k: special_quotas_map[k]
    )

    st.markdown("---")
    st.markdown("### 🎯 Preferences")

    all_branches = get_available_branches(cutoff_df)
    preferred_branches = st.multiselect(
        "Preferred Branches (in priority order)",
        options=all_branches,
        help="Select branches in order of preference. First selected = highest priority."
    )

    branch_priority = st.radio(
        "Priority Mode",
        ["Branch First", "College First"],
        help="Branch First: Best colleges for your chosen branches. College First: Top colleges regardless of branch."
    ) == "Branch First"

    college_type_options = [
        "Government", "Government Autonomous",
        "Government-Aided", "Government-Aided Autonomous",
        "Un-Aided", "Un-Aided Autonomous",
        "University Department", "University Managed",
    ]
    college_types = st.multiselect(
        "College Types to Include",
        options=college_type_options,
        default=["Government", "Government Autonomous", "Government-Aided", "Government-Aided Autonomous"],
    )

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    cap_round = st.selectbox(
        "Target CAP Round",
        options=[1, 2, 3],
        format_func=lambda x: f"CAP Round {x}"
    )

    st.markdown("**Cutoff Trend Adjustment**")
    st.caption("Adjust if you expect cutoffs to rise or fall this year vs historical average.")
    trend_adj = st.slider(
        "Percentile Points Adjustment",
        min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
        format="%.1f",
        help="Positive = expect cutoffs to rise (harder). Negative = expect cutoffs to fall (easier)."
    )
    if trend_adj > 0:
        st.caption(f"⬆️ Cutoffs {trend_adj:+.1f} pts harder than history")
    elif trend_adj < 0:
        st.caption(f"⬇️ Cutoffs {trend_adj:+.1f} pts easier than history")
    else:
        st.caption("➡️ Same as historical average")

    max_pref = st.slider("Max Preferences to Generate", 5, 20, 10)

    run_btn = st.button("🔍 Generate Recommendations", type="primary", use_container_width=True)


# ── Main Header ───────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("# 🎓 MHT-CET College Preference Advisor")
    if not cutoff_df.empty:
        years = sorted(cutoff_df['year'].unique())
        rounds = sorted(cutoff_df['cap_round'].unique())
        st.caption(
            f"📊 Loaded {len(cutoff_df):,} data points | "
            f"Years: {', '.join(str(y) for y in years)} | "
            f"Rounds: {', '.join(str(r) for r in rounds)} | "
            f"{cutoff_df['college_name'].nunique()} colleges, "
            f"{cutoff_df['course_name'].nunique()} branches"
        )
    else:
        st.warning("⚠️ No cutoff data loaded. Please place Excel files in `data/cutoffs/`.")

st.markdown("---")

# ── Data missing guard ─────────────────────────────────────────────────────────
if cutoff_df.empty:
    st.error("""
    ### No Data Found
    Please ensure your cutoff Excel files are placed in the `data/cutoffs/` folder.

    **Expected filename format:** `2022_CAP1_MH.xlsx`, `2023_CAP2_MH.xlsx`, etc.

    The files should have the same structure as the sample file provided.
    """)
    st.stop()

# ── Session State ──────────────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = pd.DataFrame()
if "preference_list" not in st.session_state:
    st.session_state.preference_list = pd.DataFrame()
if "student_profile" not in st.session_state:
    st.session_state.student_profile = {}

# ── Run predictions ────────────────────────────────────────────────────────────
if run_btn:
    if not preferred_branches:
        st.warning("⚠️ No branches selected — showing results for all branches.")

    with st.spinner("Analysing historical data and computing probabilities…"):
        predictions = generate_all_predictions(
            cutoff_df=cutoff_df,
            seat_matrix_df=seat_matrix_df,
            student_percentile=percentile,
            base_category=category,
            gender=gender,
            home_university=home_university,
            special_quotas=special_quotas,
            preferred_branches=preferred_branches,
            college_type_filter=college_types,
            target_round=cap_round,
            trend_adjustment=trend_adj,
            branch_priority=branch_priority,
            university_map=university_map,
        )
        pref_list = generate_preference_list(predictions, max_list=max_pref)

    st.session_state.predictions = predictions
    st.session_state.preference_list = pref_list
    st.session_state.student_profile = {
        "percentile": percentile,
        "category": f"{category} — {categories_map.get(category, '')}",
        "gender": gender,
        "district": district,
        "home_university": home_university,
        "special_quotas": special_quotas,
        "branches": preferred_branches,
        "cap_round": cap_round,
        "trend_adj": trend_adj,
    }
    st.success(f"✅ Found {len(predictions)} matching options across {predictions['college_name'].nunique() if not predictions.empty else 0} colleges.")


predictions = st.session_state.predictions
pref_list = st.session_state.preference_list
student_profile = st.session_state.student_profile

# ── Summary Stats (always show if data exists) ─────────────────────────────────
if not predictions.empty:
    labels = predictions['classification'].apply(lambda x: x['label'] if isinstance(x, dict) else x)
    c1, c2, c3, c4, c5 = st.columns(5)
    stat_style = lambda color, label, val: f"""
        <div class="stat-card" style="border-left-color:{color}">
            <h3>{val}</h3><p>{label}</p>
        </div>"""
    with c1:
        st.markdown(stat_style("#EF4444", "🎯 Reach", (labels == 'Reach').sum()), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_style("#F97316", "⭐ Dream", (labels == 'Dream').sum()), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_style("#3B82F6", "✅ Target", (labels == 'Target').sum()), unsafe_allow_html=True)
    with c4:
        st.markdown(stat_style("#22C55E", "🛡️ Safe", (labels == 'Safe').sum()), unsafe_allow_html=True)
    with c5:
        st.markdown(stat_style("#6B7280", "🔒 Assured", (labels == 'Assured').sum()), unsafe_allow_html=True)
    st.markdown("")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Preference List",
    "🔍 All Options",
    "📊 Round Analysis",
    "⚖️ Float / Freeze",
    "🏛️ ACAP Guide",
    "📤 Export",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREFERENCE LIST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Your Optimised CAP Preference List</div>', unsafe_allow_html=True)

    if pref_list.empty:
        st.info("👈 Fill in your profile on the left and click **Generate Recommendations**.")
    else:
        # Show profile summary
        sp = student_profile
        st.markdown(
            f"**Student:** {sp.get('percentile','—')} percentile | "
            f"Category: {sp.get('category','—')} | "
            f"District: {sp.get('district','—')} | "
            f"CAP Round {sp.get('cap_round','—')} | "
            f"Trend Adj: {sp.get('trend_adj',0):+.1f} pts"
        )
        st.markdown("")

        BADGE_COLORS = {
            "Reach": "#EF4444", "Dream": "#F97316",
            "Target": "#3B82F6", "Safe": "#22C55E", "Assured": "#6B7280"
        }

        for idx, row in pref_list.iterrows():
            cl = row['classification']
            label = cl['label'] if isinstance(cl, dict) else cl
            emoji = cl.get('emoji', '') if isinstance(cl, dict) else ''
            color = BADGE_COLORS.get(label, "#999")
            prob = row.get('probability', 0)
            cutoff = row.get('predicted_cutoff', 0)
            gap = row.get('gap', 0)
            trend = row.get('trend', 'stable')
            trend_icon = {"rising": "📈", "falling": "📉", "stable": "➡️"}.get(trend, "")
            hu = "🏠 HU" if row.get('is_home_university') else ""

            col_num, col_info, col_stats, col_badge = st.columns([0.5, 5, 2.5, 1.5])
            with col_num:
                st.markdown(f"<div style='text-align:center;font-size:1.3rem;font-weight:800;color:#1E3A5F;padding-top:4px'>{idx}</div>", unsafe_allow_html=True)
            with col_info:
                st.markdown(f"**{row['college_name']}** {hu}")
                st.caption(f"📚 {row['course_name']}  |  🏷️ {row.get('best_category','')}  |  {row.get('status','')}")
            with col_stats:
                st.markdown(f"Predicted cutoff: **{cutoff:.2f}**  {trend_icon}")
                gap_color = "#22C55E" if gap >= 0 else "#EF4444"
                st.markdown(f"Gap: <span style='color:{gap_color};font-weight:700'>{gap:+.2f} pts</span>  |  Prob: **{prob:.0f}%**", unsafe_allow_html=True)
            with col_badge:
                st.markdown(
                    f"<div style='text-align:center;padding-top:8px'>"
                    f"<span class='badge' style='background:{color}'>{emoji} {label}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            st.markdown("<hr style='margin:4px 0;border:none;border-top:1px solid #EEE'>", unsafe_allow_html=True)

        st.markdown("")
        st.info(
            "💡 **Tip:** This list is ordered for optimal strategy. "
            "Dream/Reach options are at the top — if you somehow get them, great! "
            "Target options form the bulk. Safe options protect you at the bottom. "
            "Do **not** rearrange unless you have strong personal reasons."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ALL OPTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">All Matching Options</div>', unsafe_allow_html=True)

    if predictions.empty:
        st.info("👈 Generate recommendations first.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_class = st.multiselect(
                "Filter by Classification",
                ["Reach", "Dream", "Target", "Safe", "Assured"],
                default=["Dream", "Target", "Safe", "Assured"],
                key="filter_class"
            )
        with fc2:
            all_branch_opts = sorted(predictions['course_name'].unique())
            filter_branch = st.multiselect("Filter by Branch", all_branch_opts, key="filter_branch")
        with fc3:
            min_prob = st.slider("Minimum Probability (%)", 0, 100, 5, key="min_prob")

        display = predictions.copy()
        labels_series = display['classification'].apply(lambda x: x['label'] if isinstance(x, dict) else x)
        if filter_class:
            display = display[labels_series.isin(filter_class)]
        if filter_branch:
            display = display[display['course_name'].isin(filter_branch)]
        display = display[display['probability'] >= min_prob]

        st.caption(f"Showing {len(display)} options")

        # Render as interactive table
        table_rows = []
        for _, row in display.iterrows():
            cl = row['classification']
            label = cl['label'] if isinstance(cl, dict) else cl
            emoji = cl.get('emoji', '') if isinstance(cl, dict) else ''
            table_rows.append({
                "College": row['college_name'],
                "Branch": row['course_name'],
                "Status": row.get('status', ''),
                "Category": row.get('best_category', ''),
                "Pred Cutoff": round(row.get('predicted_cutoff', 0), 2),
                "Probability": f"{row.get('probability',0):.0f}%",
                "Classification": f"{emoji} {label}",
                "Trend": {"rising": "📈 Rising", "falling": "📉 Falling", "stable": "➡️ Stable"}.get(row.get('trend',''), ""),
                "HU": "✅" if row.get('is_home_university') else "",
                "Yrs Data": row.get('data_years', 0),
            })

        if table_rows:
            tdf = pd.DataFrame(table_rows)
            st.dataframe(tdf, use_container_width=True, height=500)

        # Probability distribution chart
        st.markdown("")
        st.markdown("**Probability Distribution**")
        fig_hist = px.histogram(
            display, x='probability', nbins=20,
            color_discrete_sequence=["#3B82F6"],
            labels={"probability": "Admission Probability (%)"},
        )
        fig_hist.update_layout(
            height=250, margin=dict(t=10, b=30),
            xaxis_title="Probability (%)", yaxis_title="Count",
            plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ROUND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Round-by-Round Probability Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "See how your admission probability changes across CAP Rounds 1, 2, and 3 "
        "for any college-branch combination."
    )

    if predictions.empty:
        st.info("👈 Generate recommendations first.")
    else:
        ra_col1, ra_col2 = st.columns(2)
        with ra_col1:
            ra_college = st.selectbox(
                "Select College",
                sorted(predictions['college_name'].unique()),
                key="ra_college"
            )
        with ra_col2:
            college_branches = predictions[predictions['college_name'] == ra_college]['course_name'].unique()
            ra_branch = st.selectbox("Select Branch", sorted(college_branches), key="ra_branch")

        # Compute probability for rounds 1, 2, 3 on the fly
        round_probs = []
        for r in [1, 2, 3]:
            from src.probability_engine import analyse_college_branch, get_relevant_categories
            college_univ = university_map.get(ra_college, "")
            eligible = get_relevant_categories(
                category, gender, home_university, college_univ, special_quotas
            )
            res = analyse_college_branch(
                cutoff_df, ra_college, ra_branch,
                eligible, percentile, r, trend_adj
            )
            if res:
                round_probs.append({
                    "Round": f"Round {r}",
                    "Probability": res['probability'],
                    "Predicted Cutoff": res['predicted_cutoff'],
                    "Classification": res['classification']['label'],
                })

        if round_probs:
            rdf = pd.DataFrame(round_probs)

            fig_round = go.Figure()
            colors_map = {"Round 1": "#EF4444", "Round 2": "#F97316", "Round 3": "#22C55E"}
            fig_round.add_trace(go.Bar(
                x=rdf['Round'],
                y=rdf['Probability'],
                marker_color=[colors_map.get(r, "#3B82F6") for r in rdf['Round']],
                text=[f"{p:.0f}%" for p in rdf['Probability']],
                textposition='outside',
            ))
            fig_round.add_hline(y=30, line_dash="dash", line_color="#F97316",
                                annotation_text="Dream threshold (30%)")
            fig_round.add_hline(y=70, line_dash="dash", line_color="#22C55E",
                                annotation_text="Safe threshold (70%)")
            fig_round.update_layout(
                title=f"{ra_branch} @ {ra_college[:40]}",
                yaxis_title="Admission Probability (%)",
                yaxis_range=[0, 110],
                height=360,
                plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
            )
            st.plotly_chart(fig_round, use_container_width=True)

            st.dataframe(rdf, use_container_width=True)

            # Advice
            if len(rdf) >= 2:
                r1_prob = rdf[rdf['Round'] == 'Round 1']['Probability'].values
                r2_prob = rdf[rdf['Round'] == 'Round 2']['Probability'].values
                if len(r1_prob) and len(r2_prob):
                    diff = float(r2_prob[0]) - float(r1_prob[0])
                    if diff > 10:
                        st.success(f"💡 **Strategy:** Your probability improves by {diff:.0f}% in Round 2 for this option. Consider listing it in your Round 2 strategy if you don't get it in Round 1.")
                    else:
                        st.info("💡 **Strategy:** Probability doesn't improve much in later rounds for this option. Prioritise it in Round 1 if it's important to you.")
        else:
            st.warning("Not enough data to compute round-wise probabilities for this selection.")

        # Heatmap of top options across rounds
        st.markdown("---")
        st.markdown("**Probability Heatmap — Top 15 Options Across Rounds**")

        heatmap_data = []
        top_15 = predictions.head(15)
        for _, row in top_15.iterrows():
            for r in [1, 2, 3]:
                from src.probability_engine import analyse_college_branch, get_relevant_categories
                college_univ = university_map.get(row['college_name'], "")
                eligible = get_relevant_categories(
                    category, gender, home_university, college_univ, special_quotas
                )
                res = analyse_college_branch(
                    cutoff_df, row['college_name'], row['course_name'],
                    eligible, percentile, r, trend_adj
                )
                if res:
                    heatmap_data.append({
                        "Option": f"{row['college_name'][:25]}… / {row['course_name'][:18]}",
                        "Round": f"R{r}",
                        "Probability": res['probability'],
                    })

        if heatmap_data:
            hdf = pd.DataFrame(heatmap_data)
            hpivot = hdf.pivot(index='Option', columns='Round', values='Probability')
            fig_heat = px.imshow(
                hpivot, color_continuous_scale="RdYlGn",
                zmin=0, zmax=100, aspect="auto",
                text_auto=".0f",
            )
            fig_heat.update_layout(
                height=500, margin=dict(t=10, b=10),
                coloraxis_colorbar_title="Prob %",
                plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FLOAT / FREEZE ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Float / Freeze Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        "After getting an allocation in a CAP round, use this tool to decide whether to "
        "**Freeze** (accept and stop), **Float** (keep current, try for better), or **Slide** (try better branch at same college)."
    )

    if predictions.empty:
        st.info("👈 Generate recommendations first, then come back here.")
    else:
        ff_col1, ff_col2 = st.columns(2)
        with ff_col1:
            ff_college = st.selectbox(
                "Your Current Allocated College",
                sorted(predictions['college_name'].unique()),
                key="ff_college"
            )
        with ff_col2:
            ff_branches = predictions[predictions['college_name'] == ff_college]['course_name'].unique()
            ff_branch = st.selectbox(
                "Your Current Allocated Branch",
                sorted(ff_branches),
                key="ff_branch"
            )

        ff_round = st.selectbox(
            "Current Round (round you got this allocation in)",
            [1, 2],
            format_func=lambda x: f"Round {x}",
            key="ff_round"
        )

        # Get current probability
        from src.probability_engine import analyse_college_branch, get_relevant_categories
        college_univ = university_map.get(ff_college, "")
        eligible = get_relevant_categories(
            category, gender, home_university, college_univ, special_quotas
        )
        current_res = analyse_college_branch(
            cutoff_df, ff_college, ff_branch,
            eligible, percentile, ff_round, trend_adj
        )

        if current_res:
            current_prob = current_res['probability']
            current_cl = current_res['classification']

            st.markdown(f"""
            **Current Allocation:**
            - College: **{ff_college}**
            - Branch: **{ff_branch}**
            - Admission Probability: **{current_prob:.0f}%** ({current_cl['emoji']} {current_cl['label']})
            - Predicted Cutoff: **{current_res['predicted_cutoff']:.2f}**
            """)

            advice = float_freeze_advice(
                ff_college, ff_branch, current_prob,
                predictions, ff_round + 1
            )

            adv = advice['advice']
            if adv == "FREEZE":
                st.markdown(f"""<div class="advice-freeze">
                    <h3>🔒 Recommendation: FREEZE</h3>
                    <p>{advice['reason']}</p>
                </div>""", unsafe_allow_html=True)
            elif adv == "FLOAT":
                st.markdown(f"""<div class="advice-float">
                    <h3>🌊 Recommendation: FLOAT</h3>
                    <p>{advice['reason']}</p>
                </div>""", unsafe_allow_html=True)
                if advice.get('top_options'):
                    st.markdown("**Better options that may open up:**")
                    for opt in advice['top_options']:
                        st.markdown(f"- {opt['college_name']} / {opt['course_name']} — {opt['probability']:.0f}% probability")
            else:  # SLIDE
                st.markdown(f"""<div class="advice-slide">
                    <h3>↔️ Recommendation: SLIDE</h3>
                    <p>{advice['reason']}</p>
                </div>""", unsafe_allow_html=True)
                if advice.get('slide_options'):
                    st.markdown("**Better branches at same college:**")
                    for opt in advice['slide_options']:
                        st.markdown(f"- {opt['course_name']} — {opt['probability']:.0f}% probability")
        else:
            st.warning("Could not compute probability for this allocation. Check if the data covers this college-branch.")

        st.markdown("---")
        st.markdown("### Understanding Float / Freeze / Slide")
        st.markdown("""
        | Option | Meaning | When to Use |
        |--------|---------|-------------|
        | 🔒 **Freeze** | Accept current seat, exit the process | Your current seat is good, or you can't risk losing it |
        | 🌊 **Float** | Keep current seat as backup, try for better options | Reasonable current seat but better options likely in next round |
        | ↔️ **Slide** | Stay at same college, try for better branch | Happy with the college, but a better branch might be available |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ACAP GUIDE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">ACAP — Autonomous College Admission Process</div>', unsafe_allow_html=True)

    st.markdown("""
    ### What is ACAP?

    After the regular CAP rounds are completed, colleges with **autonomous status** conduct their own admission process for remaining vacant seats. This is called the **Autonomous College Admission Process (ACAP)** or sometimes the **Institute-Level Round**.

    ---

    ### Who Participates?

    Only colleges with **Autonomous** status are eligible to conduct ACAP. This includes:
    - Government Autonomous colleges (e.g., COEP, VJTI, SGGS Nanded)
    - Government-Aided Autonomous colleges
    - Un-Aided Autonomous colleges

    > 💡 You can identify them by **"Autonomous"** in their status in the All Options tab.

    ---

    ### How Does ACAP Work?

    1. **After CAP Round 3**, autonomous colleges receive a list of vacant seats
    2. Each college sets its own **schedule** (usually October–November)
    3. Students who were **not satisfied** with CAP allocations or **did not get any seat** can apply
    4. Merit is still based on your **MHT-CET percentile** — no new exam
    5. Categories and reservations follow the same Maharashtra rules
    6. Seats are typically very limited — mostly unfilled reserved category seats

    ---

    ### ACAP Strategy Tips

    | Tip | Detail |
    |-----|--------|
    | 📅 Track college websites | Autonomous colleges publish ACAP schedules independently |
    | 🎯 Focus on realistic options | ACAP cutoffs are often similar to or slightly lower than CAP Round 3 |
    | 📋 Keep documents ready | Same documents needed as for CAP rounds |
    | ⚡ Act quickly | ACAP windows are short — typically 2–4 days per college |
    | 🔄 ACAP doesn't cancel your seat | You can hold a CAP seat while attending ACAP counselling |

    ---

    ### Key Dates (Typical Academic Year)

    | Event | Approximate Timing |
    |-------|-------------------|
    | CAP Round 1 | August (1st week) |
    | CAP Round 2 | August (2nd–3rd week) |
    | CAP Round 3 | September (1st week) |
    | ACAP / Institute Round | September–October |
    | Classes Begin | November |

    > ⚠️ Dates change every year. Always check **fe2025.mahacet.org** for official schedules.

    ---

    ### Major Autonomous Colleges in Maharashtra

    | College | Location | Known For |
    |---------|----------|-----------|
    | COEP Technological University | Pune | CS, Mech, E&TC |
    | VJTI Mumbai | Mumbai | CS, IT, Electronics |
    | SGGS Institute of Engineering | Nanded | CS, Mech |
    | Government College of Engineering | Amravati, Karad, Aurangabad | Regional top colleges |
    | KJ Somaiya College of Engineering | Mumbai | CS, IT |
    | Pune Institute of Computer Technology | Pune | CS, IT |
    | Cummins College of Engineering | Pune | Women's college, CS, IT |
    | Symbiosis Institute of Technology | Pune | CS, IT |

    ---

    ### Important: ACAP vs CAP

    > In ACAP, you are **not protected by merit + preference order** the same way as CAP. It is more of a direct counselling process. Go prepared with your percentile certificate, category certificate, and domicile certificate.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Export Your Preference List</div>', unsafe_allow_html=True)

    if pref_list.empty:
        st.info("👈 Generate recommendations first, then export here.")
    else:
        st.markdown("### Preview")
        preview_cols = ['college_name', 'course_name', 'best_category', 'predicted_cutoff', 'probability']
        preview = pref_list[[c for c in preview_cols if c in pref_list.columns]].copy()
        preview.columns = ['College', 'Branch', 'Category', 'Pred Cutoff', 'Probability (%)']
        preview['Probability (%)'] = preview['Probability (%)'].apply(lambda x: f"{x:.0f}%")
        preview['Pred Cutoff'] = preview['Pred Cutoff'].apply(lambda x: f"{x:.2f}")
        st.dataframe(preview, use_container_width=True)

        st.markdown("---")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("### 📄 PDF Export")
            st.markdown("Download a formatted PDF with colour coding, legend, and student profile.")
            if st.button("Generate PDF", type="primary"):
                with st.spinner("Generating PDF…"):
                    try:
                        pdf_bytes = generate_pdf(student_profile, pref_list)
                        st.download_button(
                            "⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"MHTCET_Preference_List_R{student_profile.get('cap_round', 1)}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

        with exp_col2:
            st.markdown("### 📊 CSV Export")
            st.markdown("Download raw data as CSV for your own analysis.")
            if not predictions.empty:
                csv_data = predictions.copy()
                csv_data['classification_label'] = csv_data['classification'].apply(
                    lambda x: x['label'] if isinstance(x, dict) else x
                )
                csv_data = csv_data.drop(columns=['classification'], errors='ignore')
                csv_str = csv_data.to_csv(index=False)
                st.download_button(
                    "⬇️ Download All Options (CSV)",
                    data=csv_str,
                    file_name="MHTCET_All_Options.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown("---")
        st.markdown("""
        > ⚠️ **Disclaimer:** This tool uses historical data (2022–2024) for guidance only.
        > Actual cutoffs may vary. Always verify with the official Maharashtra CET Cell website
        > at **[fe2025.mahacet.org](https://fe2025.mahacet.org)** before finalising your choices.
        """)
