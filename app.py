import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import io
import copy

# --- 1. CONSTANTS & DATA MODELS ---

class Constants:
    """Holds constants for the application to improve readability and maintainability."""
    MONTHS_IN_YEAR = 12
    CPM_DIVISOR = 1000
    DEFAULT_YOUTUBE_SHARE = 0.45

class AudioInputs(BaseModel):
    """Inputs for the Audio Podcast Revenue Model"""
    monthly_downloads: int = Field(100000, description="Average monthly downloads for new episodes (used for simple projection).")
    pct_us: float = Field(0.60, description="Percentage of downloads from the US.")
    ad_load_pre: int = Field(1, description="Number of pre-roll ad slots.")
    ad_load_mid: int = Field(2, description="Number of mid-roll ad slots.")
    ad_load_post: int = Field(1, description="Number of post-roll ad slots.")
    sell_through_rate: float = Field(0.75, description="Percentage of ad inventory that is sold.")
    direct_rpm: int = Field(25, description="RPM for directly sold ads.")
    programmatic_rpm: int = Field(15, description="RPM for programmatically sold ads.")
    direct_programmatic_split: float = Field(0.80, description="Split between direct vs. programmatic ads (e.g., 0.8 = 80% direct).")
    monthly_growth_pct: float = Field(0.02, description="Expected month-over-month growth in downloads (used for simple projection).")
    manual_monthly_downloads: Optional[List[int]] = Field(None, description="A list of 12 specific download numbers for manual override.")

class YouTubeInputs(BaseModel):
    """Inputs for the YouTube Channel Revenue Model"""
    monthly_views: int = Field(500000, description="Average monthly views on new videos (used for simple projection).")
    pct_monetizable_views: float = Field(0.85, description="Percentage of views that are monetizable.")
    adsense_rpm: int = Field(8, description="Effective RPM from YouTube AdSense.")
    monthly_growth_pct: float = Field(0.03, description="Expected month-over-month growth in views (used for simple projection).")
    manual_monthly_views: Optional[List[int]] = Field(None, description="A list of 12 specific view numbers for manual override.")

class OtherRevenue(BaseModel):
    """Inputs for other cross-channel revenue streams"""
    subscriptions_monthly: int = Field(5000, description="e.g., Patreon, YouTube Memberships")
    affiliate_monthly: int = Field(1000, description="Affiliate link income.")
    other_monthly: int = Field(500, description="e.g., Merch, one-off sponsorships.")

class Costs(BaseModel):
    """Inputs for fixed and variable costs"""
    fixed_monthly: int = Field(2000, description="Monthly fixed costs (salaries, software, hosting).")
    variable_pct_gross: float = Field(0.15, description="Variable costs as a percentage of gross revenue (editing, commissions, etc.).")

class Splits(BaseModel):
    """Revenue sharing splits"""
    podcast_platform_fee_pct: float = Field(0.0, description="Fee taken by podcast hosting/distribution platform.")
    youtube_platform_fee_pct: float = Field(Constants.DEFAULT_YOUTUBE_SHARE, description="YouTube's standard ad revenue share.")
    agency_fee_pct: float = Field(0.10, description="Fee for any agency representing the creator.")
    creator_share_pct: float = Field(0.80, description="Creator's final take of the net revenue after all other splits.")


# --- 2. CALCULATION ENGINE ---

@st.cache_data
def run_annual_projection(_audio: dict, _youtube: dict, _other: dict, _costs: dict, _splits: dict) -> pd.DataFrame:
    """
    Runs a 12-month financial projection.
    Accepts dicts to be compatible with Streamlit's caching, then validates into Pydantic models.
    """
    # Convert dicts to Pydantic models inside the function for validation
    audio = AudioInputs.model_validate(_audio)
    youtube = YouTubeInputs.model_validate(_youtube)
    other = OtherRevenue.model_validate(_other)
    costs = Costs.model_validate(_costs)
    splits = Splits.model_validate(_splits)

    months = range(1, Constants.MONTHS_IN_YEAR + 1)
    data = []

    for i, month in enumerate(months):
        if audio.manual_monthly_downloads and len(audio.manual_monthly_downloads) == Constants.MONTHS_IN_YEAR:
            current_downloads = audio.manual_monthly_downloads[i]
        else:
            current_downloads = audio.monthly_downloads * ((1 + audio.monthly_growth_pct) ** i)

        if youtube.manual_monthly_views and len(youtube.manual_monthly_views) == Constants.MONTHS_IN_YEAR:
            current_views = youtube.manual_monthly_views[i]
        else:
            current_views = youtube.monthly_views * ((1 + youtube.monthly_growth_pct) ** i)

        total_ad_slots = audio.ad_load_pre + audio.ad_load_mid + audio.ad_load_post
        eligible_impressions = current_downloads * audio.pct_us * total_ad_slots
        filled_impressions = eligible_impressions * audio.sell_through_rate
        direct_impressions = filled_impressions * audio.direct_programmatic_split
        programmatic_impressions = filled_impressions * (1 - audio.direct_programmatic_split)
        direct_rev = (direct_impressions / Constants.CPM_DIVISOR) * audio.direct_rpm
        programmatic_rev = (programmatic_impressions / Constants.CPM_DIVISOR) * audio.programmatic_rpm
        gross_audio_rev = direct_rev + programmatic_rev

        monetizable_views = current_views * youtube.pct_monetizable_views
        gross_youtube_rev = (monetizable_views / Constants.CPM_DIVISOR) * youtube.adsense_rpm

        gross_other_rev = other.subscriptions_monthly + other.affiliate_monthly + other.other_monthly
        total_gross_revenue = gross_audio_rev + gross_youtube_rev + gross_other_rev

        podcast_platform_fee = gross_audio_rev * splits.podcast_platform_fee_pct
        youtube_platform_fee = gross_youtube_rev * splits.youtube_platform_fee_pct
        revenue_after_platform = total_gross_revenue - podcast_platform_fee - youtube_platform_fee
        variable_costs = total_gross_revenue * costs.variable_pct_gross
        total_costs = costs.fixed_monthly + variable_costs
        revenue_after_costs = revenue_after_platform - total_costs
        agency_fee = revenue_after_costs * splits.agency_fee_pct if revenue_after_costs > 0 else 0
        distributable_revenue = revenue_after_costs - agency_fee
        creator_net_revenue = distributable_revenue * splits.creator_share_pct if distributable_revenue > 0 else 0

        data.append({
            "Month": month,
            "Podcast Downloads": current_downloads,
            "YouTube Views": current_views,
            "Gross Audio Revenue": gross_audio_rev,
            "Gross YouTube Revenue": gross_youtube_rev,
            "Gross Other Revenue": gross_other_rev,
            "Total Gross Revenue": total_gross_revenue,
            "Platform Fees": podcast_platform_fee + youtube_platform_fee,
            "Total Business Costs": total_costs,
            "Agency Fees": agency_fee,
            "Distributable Revenue": distributable_revenue,
            "Creator Net Revenue": creator_net_revenue,
        })

    return pd.DataFrame(data)


# --- 3. STREAMLIT UI ---

def get_serializable_inputs(state):
    """Converts Pydantic models in session state to serializable dicts for caching."""
    return {
        '_audio': state.inputs['audio'].model_dump(),
        '_youtube': state.inputs['youtube'].model_dump(),
        '_other': state.inputs['other'].model_dump(),
        '_costs': state.inputs['costs'].model_dump(),
        '_splits': state.inputs['splits'].model_dump(),
    }

st.set_page_config(layout="wide", page_title="Creator Revenue Modeler")
st.title("🎙️🎬 Podcast + YouTube Revenue Modeler")
st.markdown("A tool to project annualized gross and net revenue for creators.")

# Initialize session state
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'audio': AudioInputs(), 'youtube': YouTubeInputs(), 'other': OtherRevenue(),
        'costs': Costs(), 'splits': Splits()
    }
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Inputs & Assumptions", "Results & Charts", "Scenario Comparison"])
st.sidebar.markdown("---")
st.sidebar.header("Data Ingestion (Stubs)")
st.sidebar.file_uploader("Upload Podcast Stats (CSV/XLSX)", type=['csv', 'xlsx'])
st.sidebar.file_uploader("Upload YouTube Stats (CSV/XLSX)", type=['csv', 'xlsx'])

# --- Page 1: Inputs & Assumptions ---
if page == "Inputs & Assumptions":
    st.header("Inputs & Assumptions")
    inputs = st.session_state.inputs

    # --- Presets ---
    presets = {
        "Custom": {},
        "Small Creator": {"monthly_downloads": 50000, "monthly_views": 100000, "fixed_monthly": 500},
        "Mid-Tier Creator": {"monthly_downloads": 250000, "monthly_views": 1000000, "fixed_monthly": 2500},
        "Large Creator": {"monthly_downloads": 1000000, "monthly_views": 5000000, "fixed_monthly": 10000}
    }
    preset_choice = st.selectbox("Load a Preset Scenario", options=list(presets.keys()))
    if preset_choice != "Custom":
        preset_values = presets[preset_choice]
        inputs['audio'].monthly_downloads = preset_values.get("monthly_downloads", inputs['audio'].monthly_downloads)
        inputs['youtube'].monthly_views = preset_values.get("monthly_views", inputs['youtube'].monthly_views)
        inputs['costs'].fixed_monthly = preset_values.get("fixed_monthly", inputs['costs'].fixed_monthly)
        st.success(f"Loaded '{preset_choice}' preset. You can now customize the values.")

    with st.expander("Audio Podcast Assumptions", expanded=True):
        use_manual_audio = st.toggle("Manually input monthly podcast downloads", key="manual_audio_toggle")
        if use_manual_audio:
            st.subheader("Manual Monthly Downloads")
            # Initialize with base value if not set or if length is wrong
            if not inputs['audio'].manual_monthly_downloads or len(inputs['audio'].manual_monthly_downloads) != 12:
                inputs['audio'].manual_monthly_downloads = [inputs['audio'].monthly_downloads] * 12
            
            months_df = pd.DataFrame({'Month': [f'Month {i+1}' for i in range(12)], 'Downloads': inputs['audio'].manual_monthly_downloads})
            edited_df = st.data_editor(months_df, hide_index=True, use_container_width=True, key="audio_editor")
            inputs['audio'].manual_monthly_downloads = edited_df['Downloads'].astype(int).tolist()
        else:
            inputs['audio'].manual_monthly_downloads = None
            a_col1, a_col2 = st.columns(2)
            inputs['audio'].monthly_downloads = a_col1.number_input("Avg Monthly Downloads", value=inputs['audio'].monthly_downloads, step=1000)
            inputs['audio'].monthly_growth_pct = a_col2.slider("Monthly Growth Rate %", 0.0, 0.2, inputs['audio'].monthly_growth_pct, 0.01)
        
        st.markdown("---")
        # Other audio inputs... (omitted for brevity, same as before)
        st.subheader("General Audio Ad Settings")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            inputs['audio'].pct_us = st.slider("US Download %", 0.0, 1.0, inputs['audio'].pct_us, 0.05, help=AudioInputs.model_fields['pct_us'].description)
            inputs['audio'].sell_through_rate = st.slider("Ad Sell-Through Rate", 0.0, 1.0, inputs['audio'].sell_through_rate, 0.05, help=AudioInputs.model_fields['sell_through_rate'].description)
            inputs['audio'].direct_programmatic_split = st.slider("Direct vs. Programmatic Split", 0.0, 1.0, inputs['audio'].direct_programmatic_split, 0.05, help=AudioInputs.model_fields['direct_programmatic_split'].description)
        with g_col2:
            inputs['audio'].direct_rpm = st.slider("Direct Sold Ad RPM ($)", 10, 100, inputs['audio'].direct_rpm, 1, help=AudioInputs.model_fields['direct_rpm'].description)
            inputs['audio'].programmatic_rpm = st.slider("Programmatic Ad RPM ($)", 5, 50, inputs['audio'].programmatic_rpm, 1, help=AudioInputs.model_fields['programmatic_rpm'].description)

        st.subheader("Ad Load")
        ad_col1, ad_col2, ad_col3 = st.columns(3)
        inputs['audio'].ad_load_pre = ad_col1.number_input("Pre-Rolls", value=inputs['audio'].ad_load_pre, min_value=0, max_value=5, step=1, help=AudioInputs.model_fields['ad_load_pre'].description)
        inputs['audio'].ad_load_mid = ad_col2.number_input("Mid-Rolls", value=inputs['audio'].ad_load_mid, min_value=0, max_value=10, step=1, help=AudioInputs.model_fields['ad_load_mid'].description)
        inputs['audio'].ad_load_post = ad_col3.number_input("Post-Rolls", value=inputs['audio'].ad_load_post, min_value=0, max_value=5, step=1, help=AudioInputs.model_fields['ad_load_post'].description)


    with st.expander("YouTube Channel Assumptions", expanded=True):
        use_manual_youtube = st.toggle("Manually input monthly YouTube views", key="manual_youtube_toggle")
        if use_manual_youtube:
            st.subheader("Manual Monthly Views")
            if not inputs['youtube'].manual_monthly_views or len(inputs['youtube'].manual_monthly_views) != 12:
                inputs['youtube'].manual_monthly_views = [inputs['youtube'].monthly_views] * 12
            
            yt_df = pd.DataFrame({'Month': [f'Month {i+1}' for i in range(12)], 'Views': inputs['youtube'].manual_monthly_views})
            edited_yt_df = st.data_editor(yt_df, hide_index=True, use_container_width=True, key="yt_editor")
            inputs['youtube'].manual_monthly_views = edited_yt_df['Views'].astype(int).tolist()
        else:
            inputs['youtube'].manual_monthly_views = None
            y_col1, y_col2 = st.columns(2)
            inputs['youtube'].monthly_views = y_col1.number_input("Avg Monthly Views", value=inputs['youtube'].monthly_views, step=10000)
            inputs['youtube'].monthly_growth_pct = y_col2.slider("Monthly View Growth Rate %", 0.0, 0.2, inputs['youtube'].monthly_growth_pct, 0.01)
        
        st.markdown("---")
        # Other YouTube inputs... (omitted for brevity, same as before)
        st.subheader("General YouTube Monetization Settings")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            inputs['youtube'].pct_monetizable_views = st.slider("Monetizable View %", 0.0, 1.0, inputs['youtube'].pct_monetizable_views, 0.05, help=YouTubeInputs.model_fields['pct_monetizable_views'].description)
        with m_col2:
            inputs['youtube'].adsense_rpm = st.slider("AdSense RPM ($)", 1, 30, inputs['youtube'].adsense_rpm, 1, help=YouTubeInputs.model_fields['adsense_rpm'].description)


    with st.expander("Other Revenue, Costs & Splits"):
        # Inputs for this section are the same...
        o_col1, o_col2, o_col3 = st.columns(3)
        with o_col1:
            st.subheader("Other Revenue")
            inputs['other'].subscriptions_monthly = st.number_input("Monthly Subscriptions ($)", value=inputs['other'].subscriptions_monthly, step=100, help=OtherRevenue.model_fields['subscriptions_monthly'].description)
            inputs['other'].affiliate_monthly = st.number_input("Monthly Affiliate ($)", value=inputs['other'].affiliate_monthly, step=50, help=OtherRevenue.model_fields['affiliate_monthly'].description)
            inputs['other'].other_monthly = st.number_input("Other Monthly Income ($)", value=inputs['other'].other_monthly, step=50, help=OtherRevenue.model_fields['other_monthly'].description)
        with o_col2:
            st.subheader("Business Costs")
            inputs['costs'].fixed_monthly = st.number_input("Fixed Monthly Costs ($)", value=inputs['costs'].fixed_monthly, step=100, help=Costs.model_fields['fixed_monthly'].description)
            inputs['costs'].variable_pct_gross = st.slider("Variable Costs (% of Gross)", 0.0, 0.5, inputs['costs'].variable_pct_gross, 0.01, help=Costs.model_fields['variable_pct_gross'].description)
        with o_col3:
            st.subheader("Revenue Splits")
            inputs['splits'].youtube_platform_fee_pct = st.slider("YouTube Platform Fee %", 0.0, 1.0, inputs['splits'].youtube_platform_fee_pct, 0.01, help=Splits.model_fields['youtube_platform_fee_pct'].description, disabled=True)
            inputs['splits'].agency_fee_pct = st.slider("Agency Fee %", 0.0, 0.3, inputs['splits'].agency_fee_pct, 0.01, help=Splits.model_fields['agency_fee_pct'].description)
            inputs['splits'].creator_share_pct = st.slider("Creator Final Share %", 0.0, 1.0, inputs['splits'].creator_share_pct, 0.05, help=Splits.model_fields['creator_share_pct'].description)


# --- Page 2: Results & Charts ---
elif page == "Results & Charts":
    st.header("Results & Charts")
    
    # --- Input Validation ---
    if st.session_state.inputs['splits'].creator_share_pct + st.session_state.inputs['splits'].agency_fee_pct > 1.0:
        st.error("Validation Error: The sum of the creator's share and the agency's fee cannot exceed 100% of distributable revenue.")
        st.stop()

    try:
        serializable_inputs = get_serializable_inputs(st.session_state)
        df = run_annual_projection(**serializable_inputs)
        annual_summary = df.sum()
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.stop()

    st.subheader("Annual Key Metrics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Total Gross Revenue", f"${annual_summary['Total Gross Revenue']:,.0f}")
    kpi_col2.metric("Total Business Costs", f"${annual_summary['Total Business Costs']:,.0f}")
    kpi_col3.metric("Distributable Revenue", f"${annual_summary['Distributable Revenue']:,.0f}")
    kpi_col4.metric("Creator Annual Net", f"${annual_summary['Creator Net Revenue']:,.0f}", delta=f"{annual_summary['Creator Net Revenue']/annual_summary['Total Gross Revenue']:.1%} of Gross" if annual_summary['Total Gross Revenue'] > 0 else "N/A")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.subheader("Revenue by Stream (Monthly)")
        chart_data = df[['Month', 'Gross Audio Revenue', 'Gross YouTube Revenue', 'Gross Other Revenue']].set_index('Month')
        st.bar_chart(chart_data)

    with c_col2:
        st.subheader("Annual Gross-to-Net Waterfall (Interactive)")
        net_to_others = annual_summary['Distributable Revenue'] - annual_summary['Creator Net Revenue']
        fig = go.Figure(go.Waterfall(
            name="Revenue", orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Gross Revenue", "Platform Fees", "Business Costs", "Agency Fees", "Co-creator/Partner Share", "Creator Net Revenue"],
            text=[f"${v:,.0f}" for v in [annual_summary['Total Gross Revenue'], -annual_summary['Platform Fees'], -annual_summary['Total Business Costs'], -annual_summary['Agency Fees'], -net_to_others, annual_summary['Creator Net Revenue']]],
            y=[annual_summary['Total Gross Revenue'], -annual_summary['Platform Fees'], -annual_summary['Total Business Costs'], -annual_summary['Agency Fees'], -net_to_others, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title="Annual Revenue Waterfall", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Monthly Breakdown")
    # Display logic is the same...
    display_df = df.copy()
    for col in display_df.columns:
        if 'Revenue' in col or 'Fees' in col or 'Costs' in col:
            display_df[col] = display_df[col].map('${:,.0f}'.format)
        elif 'Downloads' in col or 'Views' in col:
            display_df[col] = display_df[col].map('{:,.0f}'.format)
    st.dataframe(display_df.set_index('Month'))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Export Results to CSV", csv, "revenue_projection.csv", "text/csv")


# --- Page 3: Scenario Comparison ---
elif page == "Scenario Comparison":
    st.header("Scenario Comparison")
    st.markdown("Save your current inputs as a named scenario, then adjust them to compare outcomes.")

    col1, _ = st.columns([1, 2])
    with col1:
        scenario_name = st.text_input("Scenario Name", "Base Case")
        if st.button("Save Current Inputs as Scenario"):
            # Deep copy is essential here to snapshot the state
            st.session_state.scenarios[scenario_name] = copy.deepcopy(st.session_state.inputs)
            st.success(f"Saved scenario: '{scenario_name}'")

    if not st.session_state.scenarios:
        st.info("No scenarios saved yet. Go to the 'Inputs' page, set your assumptions, then come back here to save them.")
    else:
        comparison_data = []
        for name, inputs_models in st.session_state.scenarios.items():
            # Convert models to dicts for the cached function
            serializable_scenario_inputs = {
                '_audio': inputs_models['audio'].model_dump(),
                '_youtube': inputs_models['youtube'].model_dump(),
                '_other': inputs_models['other'].model_dump(),
                '_costs': inputs_models['costs'].model_dump(),
                '_splits': inputs_models['splits'].model_dump(),
            }
            df = run_annual_projection(**serializable_scenario_inputs)
            annual_summary = df.sum()
            comparison_data.append({
                "Scenario": name,
                "Annual Gross Revenue": annual_summary['Total Gross Revenue'],
                "Annual Creator Net": annual_summary['Creator Net Revenue'],
                "Effective Net Margin": (annual_summary['Creator Net Revenue'] / annual_summary['Total Gross Revenue']) if annual_summary['Total Gross Revenue'] > 0 else 0
            })

        comparison_df = pd.DataFrame(comparison_data).set_index("Scenario")
        st.subheader("Scenario Summary")
        st.dataframe(comparison_df.style.format({
            "Annual Gross Revenue": "${:,.0f}",
            "Annual Creator Net": "${:,.0f}",
            "Effective Net Margin": "{:.2%}"
        }))
