"""
Core calculation engine and data models for the Creator Revenue Intelligence & Modeling Tool.
This module is UI-agnostic (no Streamlit dependency) to enable unit testing and reuse.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
import warnings


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
    monthly_downloads: int = Field(100000, ge=1000, le=Constants.MAX_DOWNLOADS,
                                   description="Average monthly downloads for new episodes (1K-50M).")
    pct_us: float = Field(0.60, ge=0.0, le=1.0, description="Percentage of downloads from the US (0-100%).")
    ad_load_pre: int = Field(1, ge=0, le=10, description="Number of pre-roll ad slots (0-10).")
    ad_load_mid: int = Field(2, ge=0, le=20, description="Number of mid-roll ad slots (0-20).")
    ad_load_post: int = Field(1, ge=0, le=10, description="Number of post-roll ad slots (0-10).")
    sell_through_rate: float = Field(0.75, ge=0.0, le=1.0, description="Percentage of ad inventory that is sold (0-100%).")
    direct_rpm: int = Field(25, ge=1, le=Constants.MAX_RPM, description="RPM for directly sold ads ($1-200).")
    programmatic_rpm: int = Field(15, ge=1, le=Constants.MAX_RPM, description="RPM for programmatically sold ads ($1-200).")
    direct_programmatic_split: float = Field(0.80, ge=0.0, le=1.0, description="Split between direct vs. programmatic ads (0-100%).")
    monthly_growth_pct: float = Field(0.02, ge=-0.5, le=Constants.MAX_GROWTH_RATE,
                                      description="Expected month-over-month growth in downloads (-50% to +50%).")
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
    monthly_views: int = Field(500000, ge=1000, le=Constants.MAX_VIEWS,
                               description="Average monthly views on new videos (1K-100M).")
    pct_monetizable_views: float = Field(0.85, ge=0.0, le=1.0, description="Percentage of views that are monetizable (0-100%).")
    adsense_rpm: int = Field(8, ge=1, le=Constants.MAX_RPM, description="Effective RPM from YouTube AdSense ($1-200).")
    monthly_growth_pct: float = Field(0.03, ge=-0.5, le=Constants.MAX_GROWTH_RATE,
                                      description="Expected month-over-month growth in views (-50% to +50%).")
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
    variable_pct_gross: float = Field(0.15, ge=0.0, le=0.8,
                                      description="Variable costs as % of gross revenue: editing, commissions (0-80%).")


class Splits(BaseModel):
    """Revenue sharing splits with validation"""
    podcast_platform_fee_pct: float = Field(0.0, ge=0.0, le=0.5, description="Podcast platform fee (0-50%).")
    youtube_platform_fee_pct: float = Field(Constants.DEFAULT_YOUTUBE_SHARE, ge=0.0, le=0.7,
                                            description="YouTube's ad revenue share (0-70%).")
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


def _calculate_monthly_values(audio: AudioInputs, youtube: YouTubeInputs, month_idx: int) -> Tuple[float, float]:
    """Calculate monthly downloads and views with growth, with bounds checking."""
    try:
        if audio.manual_monthly_downloads and len(audio.manual_monthly_downloads) == Constants.MONTHS_IN_YEAR:
            current_downloads = sanitize_numeric_input(
                audio.manual_monthly_downloads[month_idx], 0, Constants.MAX_DOWNLOADS
            )
        else:
            # Prevent exponential overflow with reasonable bounds
            growth_factor = max(0.1, min(10.0, (1 + audio.monthly_growth_pct) ** month_idx))
            current_downloads = sanitize_numeric_input(
                audio.monthly_downloads * growth_factor, 0, Constants.MAX_DOWNLOADS
            )

        if youtube.manual_monthly_views and len(youtube.manual_monthly_views) == Constants.MONTHS_IN_YEAR:
            current_views = sanitize_numeric_input(
                youtube.manual_monthly_views[month_idx], 0, Constants.MAX_VIEWS
            )
        else:
            growth_factor = max(0.1, min(10.0, (1 + youtube.monthly_growth_pct) ** month_idx))
            current_views = sanitize_numeric_input(
                youtube.monthly_views * growth_factor, 0, Constants.MAX_VIEWS
            )

        return current_downloads, current_views
    except (ValueError, OverflowError, TypeError) as e:
        warnings.warn(f"Data calculation warning: using safe defaults. Error: {str(e)}")
        return 0.0, 0.0


def _calculate_audio_revenue(downloads: float, audio: AudioInputs) -> float:
    """Calculate audio/podcast revenue with bounds checking."""
    try:
        safe_downloads = sanitize_numeric_input(downloads, 0, Constants.MAX_DOWNLOADS)

        total_ad_slots = sanitize_numeric_input(
            audio.ad_load_pre + audio.ad_load_mid + audio.ad_load_post, 0, 50  # Reasonable max ad slots
        )

        eligible_impressions = sanitize_numeric_input(
            safe_downloads * audio.pct_us * total_ad_slots, 0, Constants.MAX_DOWNLOADS * 50
        )

        filled_impressions = sanitize_numeric_input(
            eligible_impressions * audio.sell_through_rate, 0, eligible_impressions
        )

        direct_impressions = sanitize_numeric_input(
            filled_impressions * audio.direct_programmatic_split, 0, filled_impressions
        )
        programmatic_impressions = filled_impressions - direct_impressions

        if Constants.CPM_DIVISOR <= 0:
            return 0.0

        direct_rev = sanitize_numeric_input(
            (direct_impressions / Constants.CPM_DIVISOR) * audio.direct_rpm, 0, 10_000_000
        )
        programmatic_rev = sanitize_numeric_input(
            (programmatic_impressions / Constants.CPM_DIVISOR) * audio.programmatic_rpm, 0, 10_000_000
        )

        return direct_rev + programmatic_rev
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as e:
        warnings.warn(f"Audio revenue calculation error: {str(e)}")
        return 0.0


def _calculate_youtube_revenue(views: float, youtube: YouTubeInputs) -> float:
    """Calculate YouTube revenue with bounds checking."""
    try:
        safe_views = sanitize_numeric_input(views, 0, Constants.MAX_VIEWS)

        monetizable_views = sanitize_numeric_input(
            safe_views * youtube.pct_monetizable_views, 0, safe_views
        )

        if Constants.CPM_DIVISOR <= 0:
            return 0.0

        return sanitize_numeric_input(
            (monetizable_views / Constants.CPM_DIVISOR) * youtube.adsense_rpm, 0, 5_000_000
        )
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as e:
        warnings.warn(f"YouTube revenue calculation error: {str(e)}")
        return 0.0


def _calculate_costs_and_splits(total_gross: float, audio_rev: float, youtube_rev: float,
                               costs: Costs, splits: Splits) -> Dict[str, float]:
    """Calculate all costs and revenue splits with bounds checking."""
    try:
        safe_total_gross = sanitize_numeric_input(total_gross, 0, 100_000_000)
        safe_audio_rev = sanitize_numeric_input(audio_rev, 0, safe_total_gross)
        safe_youtube_rev = sanitize_numeric_input(youtube_rev, 0, safe_total_gross)

        podcast_fee = sanitize_numeric_input(
            safe_audio_rev * splits.podcast_platform_fee_pct, 0, safe_audio_rev
        )
        youtube_fee = sanitize_numeric_input(
            safe_youtube_rev * splits.youtube_platform_fee_pct, 0, safe_youtube_rev
        )
        total_platform_fees = podcast_fee + youtube_fee

        variable_costs = sanitize_numeric_input(
            safe_total_gross * costs.variable_pct_gross, 0, safe_total_gross
        )
        total_costs = sanitize_numeric_input(
            costs.fixed_monthly + variable_costs, 0, 50_000_000
        )

        revenue_after_platform = max(0, safe_total_gross - total_platform_fees)
        revenue_after_costs = revenue_after_platform - total_costs

        agency_fee = 0
        if revenue_after_costs > 0:
            agency_fee = sanitize_numeric_input(
                revenue_after_costs * splits.agency_fee_pct, 0, revenue_after_costs
            )

        distributable_revenue = max(0, revenue_after_costs - agency_fee)
        creator_net_revenue = 0
        if distributable_revenue > 0:
            creator_net_revenue = sanitize_numeric_input(
                distributable_revenue * splits.creator_share_pct, 0, distributable_revenue
            )

        return {
            'platform_fees': total_platform_fees,
            'total_costs': total_costs,
            'agency_fee': agency_fee,
            'distributable_revenue': distributable_revenue,
            'creator_net_revenue': creator_net_revenue
        }
    except (ValueError, OverflowError, TypeError) as e:
        warnings.warn(f"Cost calculation error: using conservative estimates. Error: {str(e)}")
        return {
            'platform_fees': 0.0,
            'total_costs': 0.0,
            'agency_fee': 0.0,
            'distributable_revenue': 0.0,
            'creator_net_revenue': 0.0,
        }


def run_annual_projection(_audio: dict, _youtube: dict, _other: dict, _costs: dict, _splits: dict) -> pd.DataFrame:
    """
    Generate 12-month financial projections for creator revenue streams.
    This is the core calculation engine that processes all revenue, cost, and split
    data to produce monthly projections.
    """
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

        current_downloads, current_views = _calculate_monthly_values(audio, youtube, i)

        gross_audio_rev = _calculate_audio_revenue(current_downloads, audio)
        gross_youtube_rev = _calculate_youtube_revenue(current_views, youtube)
        total_gross_revenue = gross_audio_rev + gross_youtube_rev + gross_other_rev

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
