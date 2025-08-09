"""
Creator Revenue Intelligence & Modeling Tool

A comprehensive Streamlit application for content creators to model and analyze
their revenue streams across podcast and YouTube platforms.

Features:
- Multi-platform revenue modeling (Podcast + YouTube)
- Scenario comparison and analysis
- Interactive data input with validation
- Export capabilities for financial planning
- Security-hardened calculations

Author: Revenue Modeling Team
Version: 2.1.0 (Optimized, core extracted)
Last Updated: 2025
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from typing import List
import io
import copy

# Import UI-agnostic core models and functions
from core import (
    Constants,
    AudioInputs,
    YouTubeInputs,
    OtherRevenue,
    Costs,
    Splits,
    run_annual_projection,
)

# Application Configuration
st.set_page_config(
    layout="wide", 
    page_title="Revenue Modeler",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# Ultra-clean, modern minimalist styling
pio.templates.default = "plotly_white"

_GLOBAL_STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root { --text:#1a1a1a; --text-light:#6b7280; --border:#e5e7eb; --bg:#ffffff; --bg-subtle:#fafafa; --accent:#3b82f6; }
  html, body, [data-testid="stAppViewContainer"] { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); }
  [data-testid="stHeader"] { display:none; height:0; }
  [data-testid="stToolbar"] { display:none; }
  [data-testid="stAppViewContainer"] > .main { padding-top:0 !important; }
  .main .block-container { padding-top:1.25rem; padding-bottom:2rem; max-width:1200px; }
  h1 { font-weight:600; font-size:2rem; letter-spacing:-0.02em; margin:0 0 0.5rem 0; color:var(--text); }
  h2, h3 { font-weight:500; letter-spacing:-0.01em; color:var(--text); margin-top:1.5rem; margin-bottom:1rem; }
  section[data-testid="stSidebar"] { background:var(--bg); border-right:1px solid var(--border); }
  footer, #MainMenu, header { visibility:hidden; height:0; }
  .stSelectbox, .stSlider, .stNumberInput { margin-bottom:1rem; }
  .stButton button { background:var(--bg); border:1px solid var(--border); border-radius:8px; color:var(--text); font-weight:500; padding:0.5rem 1rem; transition:all .2s; }
  .stButton button:hover { border-color:var(--accent); color:var(--accent); }
  .stButton button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:#fff; }
  div[data-testid="stMetricValue"] { font-size:1.75rem; font-weight:600; color:var(--text); }
  div[data-testid="stMetricDelta"] { font-size:.875rem; font-weight:500; color:var(--text-light); }
  .streamlit-expanderHeader { font-weight:500; color:var(--text); }
  .stDataEditor { border:1px solid var(--border); border-radius:8px; }
  .stAlert { border-radius:8px; border:none; padding:1rem; }
  .element-container { margin-bottom:1rem; }
</style>
"""

# Use components.html to ensure CSS is not escaped on Streamlit Cloud
components.html(_GLOBAL_STYLE, height=0)

@st.cache_data(ttl=300)
def cached_run_annual_projection(_audio: dict, _youtube: dict, _other: dict, _costs: dict, _splits: dict) -> pd.DataFrame:
    """Thin cache wrapper around core.run_annual_projection for UI performance."""
    return run_annual_projection(_audio, _youtube, _other, _costs, _splits)


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

def create_manual_input_editor(data_type: str, values: List[int], base_value: int, key: str) -> List[int]:
    """
    Create an interactive data editor for manual monthly input values.
    
    This reusable UI component allows users to input specific monthly values
    instead of using growth projections. Includes validation and formatting.
    
    Args:
        data_type (str): Type of data (e.g., 'Downloads', 'Views')
        values (List[int]): Current monthly values (if any)
        base_value (int): Default value to use if no current values
        key (str): Unique Streamlit key for the data editor
        
    Returns:
        List[int]: List of 12 monthly values as entered by user
        
    Features:
        - Auto-fills with base_value if no existing data
        - Validates input ranges based on data type
        - Formatted number columns with reasonable limits
    """
    st.subheader(f"Manual Monthly {data_type}")
    
    # Initialize with base value if needed
    if not values or len(values) != 12:
        values = [base_value] * 12
    
    # Create DataFrame for editing
    months_df = pd.DataFrame({
        'Month': [f'Month {i+1}' for i in range(12)], 
        data_type: values
    })
    
    # Configure data editor with appropriate limits
    max_value = Constants.MAX_DOWNLOADS if 'Downloads' in data_type else Constants.MAX_VIEWS
    
    edited_df = st.data_editor(
        months_df, 
        hide_index=True, 
        use_container_width=True, 
        key=key,
        column_config={
            data_type: st.column_config.NumberColumn(
                min_value=0,
                max_value=max_value,
                format="%d"
            )
        }
    )
    
    return edited_df[data_type].astype(int).tolist()

def apply_preset(preset_name: str, inputs: dict) -> bool:
    """
    Apply predefined creator tier presets to input parameters.
    
    This function updates input parameters with realistic values for different
    creator tiers, making it easier for users to get started with appropriate
    baseline assumptions.
    
    Args:
        preset_name (str): Name of the preset tier to apply
        inputs (dict): Input parameter dictionary to update
        
    Returns:
        bool: True if preset was applied successfully, False otherwise
        
    Available Presets:
        - Small Creator: 50K downloads, 100K views, moderate RPM rates
        - Mid-Tier Creator: 250K downloads, 1M views, good RPM rates
        - Large Creator: 1M downloads, 5M views, premium RPM rates
        
    Note:
        Presets are based on industry benchmarks and typical creator metrics.
        Users should adjust values based on their specific circumstances.
    """
    # Industry-researched preset configurations
    presets = {
        "Small Creator": {
            "audio_downloads": 50000, "youtube_views": 100000, "fixed_costs": 500,
            "direct_rpm": 20, "programmatic_rpm": 12, "adsense_rpm": 6
        },
        "Mid-Tier Creator": {
            "audio_downloads": 250000, "youtube_views": 1000000, "fixed_costs": 2500,
            "direct_rpm": 30, "programmatic_rpm": 18, "adsense_rpm": 10
        },
        "Large Creator": {
            "audio_downloads": 1000000, "youtube_views": 5000000, "fixed_costs": 10000,
            "direct_rpm": 40, "programmatic_rpm": 25, "adsense_rpm": 15
        }
    }
    
    if preset_name in presets:
        preset = presets[preset_name]
        try:
            # Apply preset values with error handling
            inputs['audio'].monthly_downloads = preset["audio_downloads"]
            inputs['youtube'].monthly_views = preset["youtube_views"]
            inputs['costs'].fixed_monthly = preset["fixed_costs"]
            inputs['audio'].direct_rpm = preset["direct_rpm"]
            inputs['audio'].programmatic_rpm = preset["programmatic_rpm"]
            inputs['youtube'].adsense_rpm = preset["adsense_rpm"]
            return True
        except Exception as e:
            st.error(f"Error applying preset '{preset_name}': {e}")
            return False
    return False

def validate_inputs_ui(inputs: dict) -> bool:
    """
    Perform comprehensive UI-friendly validation of user inputs.
    
    This function checks user inputs for common mistakes and unrealistic
    values, providing helpful warnings to improve model accuracy.
    
    Args:
        inputs (dict): Dictionary containing all input models
        
    Returns:
        bool: True if all validations pass, False if warnings were issued
        
    Validation Checks:
        - Revenue split percentages don't exceed 100%
        - RPM values are reasonable and properly ordered
        - Growth rates are within sustainable ranges
        - Cost percentages are reasonable
        
    Note:
        This validation is advisory - users can still proceed with
        warnings, but should review their assumptions.
    """
    errors = []
    warnings = []
    
    # Critical validation errors
    splits = inputs['splits']
    if splits.creator_share_pct + splits.agency_fee_pct > 1.0:
        errors.append("Creator share + agency fee cannot exceed 100% of distributable revenue")
    
    # Advisory warnings for best practices
    audio = inputs['audio']
    if audio.direct_rpm <= audio.programmatic_rpm:
        warnings.append("Direct sold RPM should typically be higher than programmatic RPM")
    
    if audio.monthly_growth_pct > 0.2:
        warnings.append("Monthly growth rates above 20% may be unrealistic for sustained periods")
        
    if audio.monthly_growth_pct < -0.1:
        warnings.append("Monthly decline rates below -10% indicate serious business issues")
        
    # Cost structure warnings
    costs = inputs['costs']
    if costs.variable_pct_gross > 0.4:
        warnings.append("Variable costs above 40% of gross revenue may indicate inefficiencies")
    
    # Display errors and warnings with appropriate styling
    if errors:
        for error in errors:
            st.error(f"❌ Critical Issue: {error}")
            
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ Advisory: {warning}")
    
    return len(errors) == 0

# === MAIN APPLICATION UI ===

# Clean, minimal header
st.markdown("<h1 style='margin-bottom: 0;'>Revenue Modeler</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6b7280; margin-bottom: 2rem; font-size: 1.1rem;'>Professional revenue modeling for content creators</p>", unsafe_allow_html=True)

# Initialize session state
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'audio': AudioInputs(), 'youtube': YouTubeInputs(), 'other': OtherRevenue(),
        'costs': Costs(), 'splits': Splits()
    }
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

# Clean navigation tabs
tab1, tab2, tab3 = st.tabs(["Setup", "Results", "Scenarios"])

# Page content based on tab selection
with tab1:  # Setup tab
    inputs = st.session_state.inputs

    # Clean preset selection
    st.markdown("### Quick Start")
    col1, col2 = st.columns([3, 1])
    with col1:
        preset_choice = st.selectbox(
            "Choose a creator tier", 
            options=["Custom", "Small Creator", "Mid-Tier Creator", "Large Creator"],
            help="Select a preset to get started quickly",
            label_visibility="collapsed"
        )
    with col2:
        if preset_choice != "Custom" and st.button("Apply", type="primary", use_container_width=True):
            if apply_preset(preset_choice, inputs):
                st.success(f"Applied {preset_choice} preset")
                st.rerun()

    # Clean section headers
    st.markdown("### Audio Podcast")
    with st.container():
        use_manual_audio = st.toggle(
            "Manual monthly input", 
            key="manual_audio_toggle",
            help="Enable for month-by-month control"
        )
        
        if use_manual_audio:
            st.markdown("**Monthly Downloads**")
            if not inputs['audio'].manual_monthly_downloads or len(inputs['audio'].manual_monthly_downloads) != 12:
                inputs['audio'].manual_monthly_downloads = [inputs['audio'].monthly_downloads] * 12
            
            # Clean 4x3 grid layout
            for quarter in range(4):
                cols = st.columns(3)
                for month_in_quarter in range(3):
                    month_idx = quarter * 3 + month_in_quarter
                    if month_idx < 12:
                        with cols[month_in_quarter]:
                            inputs['audio'].manual_monthly_downloads[month_idx] = st.number_input(
                                f"Month {month_idx + 1}",
                                value=inputs['audio'].manual_monthly_downloads[month_idx],
                                min_value=0,
                                step=1000,
                                key=f"audio_month_{month_idx}",
                                label_visibility="visible"
                            )
        else:
            inputs['audio'].manual_monthly_downloads = None
            col1, col2 = st.columns(2)
            with col1:
                inputs['audio'].monthly_downloads = st.number_input(
                    "Monthly Downloads", 
                    value=inputs['audio'].monthly_downloads, 
                    step=1000, min_value=1000,
                    help="Average downloads per month"
                )
            with col2:
                inputs['audio'].monthly_growth_pct = st.slider(
                    "Growth Rate %", 
                    -20, 20, 
                    int(inputs['audio'].monthly_growth_pct * 100),
                    help="Month-over-month growth"
                ) / 100
        
        st.markdown("---")
        # Other audio inputs... (omitted for brevity, same as before)
        # Clean, simple inputs
        col1, col2 = st.columns(2)
        with col1:
            inputs['audio'].pct_us = st.slider("US Downloads %", 0, 100, int(inputs['audio'].pct_us*100), help="Percentage from US market") / 100
            inputs['audio'].sell_through_rate = st.slider("Sell-through Rate %", 0, 100, int(inputs['audio'].sell_through_rate*100)) / 100
            inputs['audio'].direct_rpm = st.number_input("Direct RPM $", 10, 100, inputs['audio'].direct_rpm, help="Revenue per thousand for direct ads")
        with col2:
            inputs['audio'].direct_programmatic_split = st.slider("Direct vs Programmatic %", 0, 100, int(inputs['audio'].direct_programmatic_split*100)) / 100
            inputs['audio'].programmatic_rpm = st.number_input("Programmatic RPM $", 5, 50, inputs['audio'].programmatic_rpm, help="Revenue per thousand for programmatic ads")
        
        st.markdown("**Ad Slots**")
        ad_col1, ad_col2, ad_col3 = st.columns(3)
        inputs['audio'].ad_load_pre = ad_col1.number_input("Pre-roll", 0, 5, inputs['audio'].ad_load_pre, help="Ads before content")
        inputs['audio'].ad_load_mid = ad_col2.number_input("Mid-roll", 0, 10, inputs['audio'].ad_load_mid, help="Ads during content")
        inputs['audio'].ad_load_post = ad_col3.number_input("Post-roll", 0, 5, inputs['audio'].ad_load_post, help="Ads after content")


    st.markdown("### YouTube Channel")
    with st.container():
        use_manual_youtube = st.toggle(
            "Manual monthly input", 
            key="manual_youtube_toggle",
            help="Enable for month-by-month control"
        )
        
        if use_manual_youtube:
            st.markdown("**Monthly Views**")
            if not inputs['youtube'].manual_monthly_views or len(inputs['youtube'].manual_monthly_views) != 12:
                inputs['youtube'].manual_monthly_views = [inputs['youtube'].monthly_views] * 12
            
            # Clean 4x3 grid layout
            for quarter in range(4):
                cols = st.columns(3)
                for month_in_quarter in range(3):
                    month_idx = quarter * 3 + month_in_quarter
                    if month_idx < 12:
                        with cols[month_in_quarter]:
                            inputs['youtube'].manual_monthly_views[month_idx] = st.number_input(
                                f"Month {month_idx + 1}",
                                value=inputs['youtube'].manual_monthly_views[month_idx],
                                min_value=0,
                                step=10000,
                                key=f"youtube_month_{month_idx}",
                                label_visibility="visible"
                            )
        else:
            inputs['youtube'].manual_monthly_views = None
            col1, col2 = st.columns(2)
            with col1:
                inputs['youtube'].monthly_views = st.number_input(
                    "Monthly Views", 
                    value=inputs['youtube'].monthly_views, 
                    step=10000, min_value=1000,
                    help="Average views per month"
                )
            with col2:
                inputs['youtube'].monthly_growth_pct = st.slider(
                    "Growth Rate %", 
                    -20, 20, 
                    int(inputs['youtube'].monthly_growth_pct * 100),
                    help="Month-over-month growth"
                ) / 100
        
        st.markdown("---")
        # Other YouTube inputs... (omitted for brevity, same as before)
        col1, col2 = st.columns(2)
        with col1:
            inputs['youtube'].pct_monetizable_views = st.slider("Monetizable Views %", 0, 100, int(inputs['youtube'].pct_monetizable_views*100)) / 100
        with col2:
            inputs['youtube'].adsense_rpm = st.number_input("AdSense RPM $", 1, 30, inputs['youtube'].adsense_rpm, help="Revenue per thousand views")


    st.markdown("### Revenue & Costs")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Other Revenue**")
            inputs['other'].subscriptions_monthly = st.number_input(
                "Subscriptions $", 
                value=inputs['other'].subscriptions_monthly, 
                step=100, min_value=0,
                help="Patreon, memberships"
            )
            inputs['other'].affiliate_monthly = st.number_input(
                "Affiliate $", 
                value=inputs['other'].affiliate_monthly, 
                step=50, min_value=0,
                help="Commission income"
            )
            inputs['other'].other_monthly = st.number_input(
                "Other Income $", 
                value=inputs['other'].other_monthly, 
                step=50, min_value=0,
                help="Merch, sponsorships"
            )
        with col2:
            st.markdown("**Business Costs**")
            inputs['costs'].fixed_monthly = st.number_input(
                "Fixed Costs $", 
                value=inputs['costs'].fixed_monthly, 
                step=100, min_value=0,
                help="Salaries, software, hosting"
            )
            inputs['costs'].variable_pct_gross = st.slider(
                "Variable Costs %", 
                0, 50, 
                int(inputs['costs'].variable_pct_gross * 100),
                help="Editing, commissions"
            ) / 100
        with col3:
            st.markdown("**Revenue Splits**")
            st.caption("YouTube fee: 45% (platform standard)")
            inputs['splits'].agency_fee_pct = st.slider(
                "Agency Fee %", 
                0, 30, 
                int(inputs['splits'].agency_fee_pct * 100),
                help="Management fees"
            ) / 100
            inputs['splits'].creator_share_pct = st.slider(
                "Creator Share %", 
                50, 100, 
                int(inputs['splits'].creator_share_pct * 100),
                help="Your final take"
            ) / 100


with tab2:  # Results tab
    st.markdown("### Revenue Analysis")
    
    # --- Enhanced Input Validation ---
    if not validate_inputs_ui(st.session_state.inputs):
        st.warning("⚠️ Please review the validation warnings above and adjust your inputs.")
        st.info("💡 **Tip:** Use the validation warnings to optimize your revenue model for better accuracy.")
    
    # --- Calculation with loading state ---
    with st.spinner('📊 Calculating your revenue projections...'):
        try:
            serializable_inputs = get_serializable_inputs(st.session_state)
            df = cached_run_annual_projection(**serializable_inputs)
            annual_summary = df.sum()
        except Exception as e:
            st.error(f"😱 Calculation error: {str(e)}")
            st.info("🔧 Please check your input values and try again. If the problem persists, try resetting to default values.")
            st.stop()

    st.markdown("### Key Metrics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    # Calculate profit margin for better insights
    profit_margin = (annual_summary['Creator Net Revenue'] / annual_summary['Total Gross Revenue']) if annual_summary['Total Gross Revenue'] > 0 else 0
    monthly_avg = annual_summary['Creator Net Revenue'] / 12 if annual_summary['Creator Net Revenue'] > 0 else 0
    
    kpi_col1.metric(
        "💰 Total Gross Revenue", 
        f"${annual_summary['Total Gross Revenue']:,.0f}",
        help="Total revenue before any deductions"
    )
    kpi_col2.metric(
        "💸 Total Business Costs", 
        f"${annual_summary['Total Business Costs']:,.0f}",
        delta=f"{(annual_summary['Total Business Costs']/annual_summary['Total Gross Revenue']*100):.1f}% of gross" if annual_summary['Total Gross Revenue'] > 0 else "N/A",
        help="All business expenses including fixed and variable costs"
    )
    kpi_col3.metric(
        "📋 Distributable Revenue", 
        f"${annual_summary['Distributable Revenue']:,.0f}",
        help="Revenue available for creator and partners after all costs"
    )
    kpi_col4.metric(
        "🎆 Creator Annual Net", 
        f"${annual_summary['Creator Net Revenue']:,.0f}",
        delta=f"{profit_margin:.1%} profit margin" if annual_summary['Total Gross Revenue'] > 0 else "N/A",
        help=f"Your final take-home: ~${monthly_avg:,.0f}/month"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Monthly Revenue by Stream**")
        months = df['Month']
        rev_fig = go.Figure()
        rev_fig.add_trace(go.Bar(name="Audio", x=months, y=df['Gross Audio Revenue'], marker_color="#1a1a1a"))
        rev_fig.add_trace(go.Bar(name="YouTube", x=months, y=df['Gross YouTube Revenue'], marker_color="#3b82f6"))
        rev_fig.add_trace(go.Bar(name="Other", x=months, y=df['Gross Other Revenue'], marker_color="#6b7280"))
        rev_fig.update_layout(
            barmode="group",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=10, b=60, l=10, r=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=12, color="#1a1a1a"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f5f5f5")
        )
        st.plotly_chart(rev_fig, use_container_width=True)

    with col2:
        st.markdown("**Revenue Breakdown**")
        net_to_others = annual_summary['Distributable Revenue'] - annual_summary['Creator Net Revenue']
        fig = go.Figure(go.Waterfall(
            name="Revenue", orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Gross", "Platform Fees", "Costs", "Agency", "Partners", "Creator Net"],
            text=[f"${v:,.0f}" for v in [annual_summary['Total Gross Revenue'], -annual_summary['Platform Fees'], -annual_summary['Total Business Costs'], -annual_summary['Agency Fees'], -net_to_others, annual_summary['Creator Net Revenue']]],
            y=[annual_summary['Total Gross Revenue'], -annual_summary['Platform Fees'], -annual_summary['Total Business Costs'], -annual_summary['Agency Fees'], -net_to_others, 0],
            connector={"line": {"color": "#e5e7eb", "width": 1}},
            increasing={"marker": {"color": "#1a1a1a"}},
            decreasing={"marker": {"color": "#6b7280"}},
            totals={"marker": {"color": "#3b82f6"}},
        ))
        fig.update_layout(
            showlegend=False, 
            margin=dict(t=10, b=40, l=10, r=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=12, color="#1a1a1a"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f5f5f5")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Breakdown")
    # Display logic is the same...
    display_df = df.copy()
    for col in display_df.columns:
        if 'Revenue' in col or 'Fees' in col or 'Costs' in col:
            display_df[col] = display_df[col].map('${:,.0f}'.format)
        elif 'Downloads' in col or 'Views' in col:
            display_df[col] = display_df[col].map('{:,.0f}'.format)
    st.dataframe(display_df.set_index('Month'))

    # Enhanced export options
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Export to CSV", 
            csv, 
            "revenue_projection.csv", 
            "text/csv",
            help="Download detailed monthly breakdown"
        )
    with export_col2:
        # Create summary export
        summary_data = {
            'Metric': ['Total Gross Revenue', 'Total Costs', 'Creator Net Revenue', 'Profit Margin'],
            'Annual Value': [f"${annual_summary['Total Gross Revenue']:,.0f}", 
                           f"${annual_summary['Total Business Costs']:,.0f}",
                           f"${annual_summary['Creator Net Revenue']:,.0f}",
                           f"{profit_margin:.1%}"]
        }
        summary_csv = pd.DataFrame(summary_data).to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Export Summary", 
            summary_csv, 
            "revenue_summary.csv", 
            "text/csv",
            help="Download key metrics summary"
        )


with tab3:  # Scenarios tab
    st.markdown("### Scenario Comparison")
    st.markdown("Save and compare different revenue strategies to optimize your approach.")

    col1, _ = st.columns([1, 2])
    with col1:
        scenario_name = st.text_input("Scenario Name", "Base Case")
        if st.button("Save Current Inputs as Scenario", type="primary"):
            if scenario_name and scenario_name.strip():
                # Deep copy is essential here to snapshot the state
                st.session_state.scenarios[scenario_name] = copy.deepcopy(st.session_state.inputs)
                st.success(f"✅ Saved scenario: '{scenario_name}'")
                st.balloons()  # Fun user feedback
            else:
                st.error("❌ Please enter a scenario name")

    if not st.session_state.scenarios:
        st.info("💭 **No scenarios saved yet.** Go to 'Inputs & Assumptions', configure your model, then return here to save and compare scenarios.")
        st.markdown("💡 **Pro tip:** Create scenarios like 'Conservative', 'Optimistic', and 'Realistic' to understand your revenue range.")
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
            df = cached_run_annual_projection(**serializable_scenario_inputs)
            annual_summary = df.sum()
            comparison_data.append({
                "Scenario": name,
                "Annual Gross Revenue": annual_summary['Total Gross Revenue'],
                "Annual Creator Net": annual_summary['Creator Net Revenue'],
                "Effective Net Margin": (annual_summary['Creator Net Revenue'] / annual_summary['Total Gross Revenue']) if annual_summary['Total Gross Revenue'] > 0 else 0
            })

        comparison_df = pd.DataFrame(comparison_data).set_index("Scenario")
        st.markdown("**Scenario Comparison**")
        st.dataframe(comparison_df.style.format({
            "Annual Gross Revenue": "${:,.0f}",
            "Annual Creator Net": "${:,.0f}",
            "Effective Net Margin": "{:.2%}"
        }))
