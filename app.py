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
Version: 2.0.0 (Optimized)
Last Updated: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional
import io
import copy

# Application Configuration
st.set_page_config(
    layout="wide", 
    page_title="Revenue Modeler",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# Ultra-clean, modern minimalist styling
pio.templates.default = "plotly_white"
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
      :root {
        --text: #1a1a1a;
        --text-light: #6b7280;
        --border: #e5e7eb;
        --bg: #ffffff;
        --bg-subtle: #fafafa;
        --accent: #3b82f6;
      }
      
      html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--bg);
        color: var(--text);
      }
      
      .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
      }
      
      /* Clean headers */
      h1 {
        font-weight: 600;
        font-size: 2rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: var(--text);
      }
      
      h2, h3 {
        font-weight: 500;
        letter-spacing: -0.01em;
        color: var(--text);
        margin-top: 2rem;
        margin-bottom: 1rem;
      }
      
      /* Clean sidebar */
      section[data-testid="stSidebar"] {
        background: var(--bg);
        border-right: 1px solid var(--border);
      }
      
      /* Hide Streamlit branding */
      footer, #MainMenu, header {
        visibility: hidden;
      }
      
      /* Clean inputs */
      .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
      }
      
      /* Minimal buttons */
      .stButton button {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
      }
      
      .stButton button:hover {
        border-color: var(--accent);
        color: var(--accent);
      }
      
      .stButton button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
      }
      
      /* Clean metrics */
      div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text);
      }
      
      div[data-testid="stMetricDelta"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-light);
      }
      
      /* Clean expanders */
      .streamlit-expanderHeader {
        font-weight: 500;
        color: var(--text);
      }
      
      /* Clean data editor */
      .stDataEditor {
        border: 1px solid var(--border);
        border-radius: 8px;
      }
      
      /* Minimal alerts */
      .stAlert {
        border-radius: 8px;
        border: none;
        padding: 1rem;
      }
      
      /* Clean spacing */
      .element-container {
        margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 1. CONSTANTS & DATA MODELS ---

class Constants:
    """Application constants for better maintainability and performance."""
    MONTHS_IN_YEAR = 12
    CPM_DIVISOR = 1000
    DEFAULT_YOUTUBE_SHARE = 0.45
    MAX_GROWTH_RATE = 0.5  # 50% max growth rate
    MAX_RPM = 200  # Max realistic RPM
    MAX_DOWNLOADS = 50_000_000  # Max realistic downloads
    MAX_VIEWS = 100_000_000  # Max realistic views

class AudioInputs(BaseModel):
    """Inputs for the Audio Podcast Revenue Model with validation"""
    monthly_downloads: int = Field(100000, ge=1000, le=Constants.MAX_DOWNLOADS, description="Average monthly downloads for new episodes (1K-50M).")
    pct_us: float = Field(0.60, ge=0.0, le=1.0, description="Percentage of downloads from the US (0-100%).")
    ad_load_pre: int = Field(1, ge=0, le=10, description="Number of pre-roll ad slots (0-10).")
    ad_load_mid: int = Field(2, ge=0, le=20, description="Number of mid-roll ad slots (0-20).")
    ad_load_post: int = Field(1, ge=0, le=10, description="Number of post-roll ad slots (0-10).")
    sell_through_rate: float = Field(0.75, ge=0.0, le=1.0, description="Percentage of ad inventory that is sold (0-100%).")
    direct_rpm: int = Field(25, ge=1, le=Constants.MAX_RPM, description="RPM for directly sold ads ($1-200).")
    programmatic_rpm: int = Field(15, ge=1, le=Constants.MAX_RPM, description="RPM for programmatically sold ads ($1-200).")
    direct_programmatic_split: float = Field(0.80, ge=0.0, le=1.0, description="Split between direct vs. programmatic ads (0-100%).")
    monthly_growth_pct: float = Field(0.02, ge=-0.5, le=Constants.MAX_GROWTH_RATE, description="Expected month-over-month growth in downloads (-50% to +50%).")
    manual_monthly_downloads: Optional[List[int]] = Field(None, description="A list of 12 specific download numbers for manual override.")
    
    @field_validator('manual_monthly_downloads')
    @classmethod
    def validate_manual_downloads(cls, v):
        if v is not None:
            if len(v) != Constants.MONTHS_IN_YEAR:
                raise ValueError(f"Manual downloads must have exactly {Constants.MONTHS_IN_YEAR} values")
            if any(d < 0 or d > Constants.MAX_DOWNLOADS for d in v):
                raise ValueError(f"Download values must be between 0 and {Constants.MAX_DOWNLOADS:,}")
        return v

class YouTubeInputs(BaseModel):
    """Inputs for the YouTube Channel Revenue Model with validation"""
    monthly_views: int = Field(500000, ge=1000, le=Constants.MAX_VIEWS, description="Average monthly views on new videos (1K-100M).")
    pct_monetizable_views: float = Field(0.85, ge=0.0, le=1.0, description="Percentage of views that are monetizable (0-100%).")
    adsense_rpm: int = Field(8, ge=1, le=Constants.MAX_RPM, description="Effective RPM from YouTube AdSense ($1-200).")
    monthly_growth_pct: float = Field(0.03, ge=-0.5, le=Constants.MAX_GROWTH_RATE, description="Expected month-over-month growth in views (-50% to +50%).")
    manual_monthly_views: Optional[List[int]] = Field(None, description="A list of 12 specific view numbers for manual override.")
    
    @field_validator('manual_monthly_views')
    @classmethod
    def validate_manual_views(cls, v):
        if v is not None:
            if len(v) != Constants.MONTHS_IN_YEAR:
                raise ValueError(f"Manual views must have exactly {Constants.MONTHS_IN_YEAR} values")
            if any(view < 0 or view > Constants.MAX_VIEWS for view in v):
                raise ValueError(f"View values must be between 0 and {Constants.MAX_VIEWS:,}")
        return v

class OtherRevenue(BaseModel):
    """Inputs for other cross-channel revenue streams with validation"""
    subscriptions_monthly: int = Field(5000, ge=0, le=1_000_000, description="Monthly subscription revenue: Patreon, memberships ($0-1M).")
    affiliate_monthly: int = Field(1000, ge=0, le=500_000, description="Monthly affiliate income ($0-500K).")
    other_monthly: int = Field(500, ge=0, le=500_000, description="Other monthly income: merch, sponsorships ($0-500K).")

class Costs(BaseModel):
    """Inputs for fixed and variable costs with validation"""
    fixed_monthly: int = Field(2000, ge=0, le=1_000_000, description="Monthly fixed costs: salaries, software, hosting ($0-1M).")
    variable_pct_gross: float = Field(0.15, ge=0.0, le=0.8, description="Variable costs as % of gross revenue: editing, commissions (0-80%).")

class Splits(BaseModel):
    """Revenue sharing splits with validation"""
    podcast_platform_fee_pct: float = Field(0.0, ge=0.0, le=0.5, description="Podcast platform fee (0-50%).")
    youtube_platform_fee_pct: float = Field(Constants.DEFAULT_YOUTUBE_SHARE, ge=0.0, le=0.7, description="YouTube's ad revenue share (0-70%).")
    agency_fee_pct: float = Field(0.10, ge=0.0, le=0.5, description="Agency fee (0-50%).")
    creator_share_pct: float = Field(0.80, ge=0.1, le=1.0, description="Creator's final share of net revenue (10-100%).")
    
    @model_validator(mode='after')
    def validate_total_splits(self):
        total_non_creator = self.agency_fee_pct + (1 - self.creator_share_pct)
        if total_non_creator > 0.9:  # Allow up to 90% to go to non-creator
            raise ValueError("Total fees and partner shares cannot exceed 90% of distributable revenue")
        return self


# --- 2. CALCULATION ENGINE ---

def sanitize_numeric_input(value: float, min_val: float = 0, max_val: float = float('inf')) -> float:
    """Sanitize numeric inputs to prevent overflow and invalid values."""
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return min_val
    return max(min_val, min(max_val, float(value)))

def _calculate_monthly_values(audio: AudioInputs, youtube: YouTubeInputs, month_idx: int) -> tuple[float, float]:
    """Calculate monthly downloads and views with growth, with security bounds checking."""
    try:
        if audio.manual_monthly_downloads and len(audio.manual_monthly_downloads) == Constants.MONTHS_IN_YEAR:
            current_downloads = sanitize_numeric_input(
                audio.manual_monthly_downloads[month_idx], 
                0, 
                Constants.MAX_DOWNLOADS
            )
        else:
            # Prevent exponential overflow with reasonable bounds
            growth_factor = max(0.1, min(10.0, (1 + audio.monthly_growth_pct) ** month_idx))
            current_downloads = sanitize_numeric_input(
                audio.monthly_downloads * growth_factor, 
                0, 
                Constants.MAX_DOWNLOADS
            )
        
        if youtube.manual_monthly_views and len(youtube.manual_monthly_views) == Constants.MONTHS_IN_YEAR:
            current_views = sanitize_numeric_input(
                youtube.manual_monthly_views[month_idx], 
                0, 
                Constants.MAX_VIEWS
            )
        else:
            # Prevent exponential overflow with reasonable bounds
            growth_factor = max(0.1, min(10.0, (1 + youtube.monthly_growth_pct) ** month_idx))
            current_views = sanitize_numeric_input(
                youtube.monthly_views * growth_factor, 
                0, 
                Constants.MAX_VIEWS
            )
        
        return current_downloads, current_views
    except (ValueError, OverflowError, TypeError) as e:
        st.warning(f"Data calculation warning: using safe defaults. Error: {str(e)}")
        return 0.0, 0.0

def _calculate_audio_revenue(downloads: float, audio: AudioInputs) -> float:
    """Calculate audio/podcast revenue with security bounds checking."""
    try:
        # Sanitize all inputs to prevent overflow attacks
        safe_downloads = sanitize_numeric_input(downloads, 0, Constants.MAX_DOWNLOADS)
        
        total_ad_slots = sanitize_numeric_input(
            audio.ad_load_pre + audio.ad_load_mid + audio.ad_load_post, 
            0, 50  # Reasonable max ad slots
        )
        
        eligible_impressions = sanitize_numeric_input(
            safe_downloads * audio.pct_us * total_ad_slots, 
            0, Constants.MAX_DOWNLOADS * 50
        )
        
        filled_impressions = sanitize_numeric_input(
            eligible_impressions * audio.sell_through_rate,
            0, eligible_impressions
        )
        
        direct_impressions = sanitize_numeric_input(
            filled_impressions * audio.direct_programmatic_split,
            0, filled_impressions
        )
        programmatic_impressions = filled_impressions - direct_impressions
        
        # Prevent division by zero and ensure reasonable CPM calculations
        if Constants.CPM_DIVISOR <= 0:
            return 0.0
            
        direct_rev = sanitize_numeric_input(
            (direct_impressions / Constants.CPM_DIVISOR) * audio.direct_rpm,
            0, 10_000_000  # Max reasonable monthly revenue
        )
        programmatic_rev = sanitize_numeric_input(
            (programmatic_impressions / Constants.CPM_DIVISOR) * audio.programmatic_rpm,
            0, 10_000_000
        )
        
        return direct_rev + programmatic_rev
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as e:
        st.warning(f"Audio revenue calculation error: {str(e)}")
        return 0.0

def _calculate_youtube_revenue(views: float, youtube: YouTubeInputs) -> float:
    """Calculate YouTube revenue with security bounds checking."""
    try:
        safe_views = sanitize_numeric_input(views, 0, Constants.MAX_VIEWS)
        
        monetizable_views = sanitize_numeric_input(
            safe_views * youtube.pct_monetizable_views,
            0, safe_views
        )
        
        if Constants.CPM_DIVISOR <= 0:
            return 0.0
            
        return sanitize_numeric_input(
            (monetizable_views / Constants.CPM_DIVISOR) * youtube.adsense_rpm,
            0, 5_000_000  # Max reasonable monthly YouTube revenue
        )
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as e:
        st.warning(f"YouTube revenue calculation error: {str(e)}")
        return 0.0

def _calculate_costs_and_splits(total_gross: float, audio_rev: float, youtube_rev: float, 
                               costs: Costs, splits: Splits) -> dict:
    """Calculate all costs and revenue splits with security bounds checking."""
    try:
        # Sanitize all revenue inputs
        safe_total_gross = sanitize_numeric_input(total_gross, 0, 100_000_000)
        safe_audio_rev = sanitize_numeric_input(audio_rev, 0, safe_total_gross)
        safe_youtube_rev = sanitize_numeric_input(youtube_rev, 0, safe_total_gross)
        
        # Platform fees with bounds checking
        podcast_fee = sanitize_numeric_input(
            safe_audio_rev * splits.podcast_platform_fee_pct,
            0, safe_audio_rev
        )
        youtube_fee = sanitize_numeric_input(
            safe_youtube_rev * splits.youtube_platform_fee_pct,
            0, safe_youtube_rev
        )
        total_platform_fees = podcast_fee + youtube_fee
        
        # Costs with bounds checking
        variable_costs = sanitize_numeric_input(
            safe_total_gross * costs.variable_pct_gross,
            0, safe_total_gross
        )
        total_costs = sanitize_numeric_input(
            costs.fixed_monthly + variable_costs,
            0, 50_000_000  # Reasonable max monthly costs
        )
        
        # Revenue after costs
        revenue_after_platform = max(0, safe_total_gross - total_platform_fees)
        revenue_after_costs = revenue_after_platform - total_costs
        
        # Final splits with security bounds
        agency_fee = 0
        if revenue_after_costs > 0:
            agency_fee = sanitize_numeric_input(
                revenue_after_costs * splits.agency_fee_pct,
                0, revenue_after_costs
            )
        
        distributable_revenue = max(0, revenue_after_costs - agency_fee)
        creator_net_revenue = 0
        if distributable_revenue > 0:
            creator_net_revenue = sanitize_numeric_input(
                distributable_revenue * splits.creator_share_pct,
                0, distributable_revenue
            )
        
        return {
            'platform_fees': total_platform_fees,
            'total_costs': total_costs,
            'agency_fee': agency_fee,
            'distributable_revenue': distributable_revenue,
            'creator_net_revenue': creator_net_revenue
        }
    except (ValueError, OverflowError, TypeError) as e:
        st.warning(f"Cost calculation error: using conservative estimates. Error: {str(e)}")
        return {
            'platform_fees': 0,
            'total_costs': 0,
            'agency_fee': 0,
            'distributable_revenue': 0,
            'creator_net_revenue': 0
        }

@st.cache_data(ttl=300)  # Cache for 5 minutes - improves performance for repeated calculations
def run_annual_projection(_audio: dict, _youtube: dict, _other: dict, _costs: dict, _splits: dict) -> pd.DataFrame:
    """
    Generate 12-month financial projections for creator revenue streams.
    
    This is the core calculation engine that processes all revenue, cost, and split
    data to produce monthly projections. Optimized for performance with modular
    functions and comprehensive error handling.
    
    Args:
        _audio (dict): Audio/podcast revenue parameters
        _youtube (dict): YouTube revenue parameters  
        _other (dict): Other revenue streams (subscriptions, affiliate, etc.)
        _costs (dict): Business cost parameters (fixed and variable)
        _splits (dict): Revenue sharing and platform fee parameters
    
    Returns:
        pd.DataFrame: Monthly financial projections with 12 rows containing:
            - Platform metrics (downloads, views)
            - Revenue streams (audio, YouTube, other)
            - Costs and fees breakdown
            - Final creator net revenue
    
    Raises:
        ValidationError: If input data fails Pydantic validation
        ValueError: If calculations result in invalid values
    """
    try:
        # Convert and validate inputs
        audio = AudioInputs.model_validate(_audio)
        youtube = YouTubeInputs.model_validate(_youtube)
        other = OtherRevenue.model_validate(_other)
        costs = Costs.model_validate(_costs)
        splits = Splits.model_validate(_splits)
        
        # Pre-calculate static values
        gross_other_rev = other.subscriptions_monthly + other.affiliate_monthly + other.other_monthly
        
        data = []
        for i in range(Constants.MONTHS_IN_YEAR):
            month = i + 1
            
            # Calculate monthly volumes
            current_downloads, current_views = _calculate_monthly_values(audio, youtube, i)
            
            # Calculate revenues
            gross_audio_rev = _calculate_audio_revenue(current_downloads, audio)
            gross_youtube_rev = _calculate_youtube_revenue(current_views, youtube)
            total_gross_revenue = gross_audio_rev + gross_youtube_rev + gross_other_rev
            
            # Calculate costs and splits
            financials = _calculate_costs_and_splits(total_gross_revenue, gross_audio_rev, 
                                                   gross_youtube_rev, costs, splits)
            
            data.append({
                "Month": month,
                "Podcast Downloads": int(current_downloads),
                "YouTube Views": int(current_views),
                "Gross Audio Revenue": gross_audio_rev,
                "Gross YouTube Revenue": gross_youtube_rev,
                "Gross Other Revenue": gross_other_rev,
                "Total Gross Revenue": total_gross_revenue,
                "Platform Fees": financials['platform_fees'],
                "Total Business Costs": financials['total_costs'],
                "Agency Fees": financials['agency_fee'],
                "Distributable Revenue": financials['distributable_revenue'],
                "Creator Net Revenue": financials['creator_net_revenue'],
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        raise


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
            df = run_annual_projection(**serializable_inputs)
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
            df = run_annual_projection(**serializable_scenario_inputs)
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
