import math

from core import (
    AudioInputs,
    YouTubeInputs,
    OtherRevenue,
    Costs,
    Splits,
    run_annual_projection,
    sanitize_numeric_input,
    _calculate_monthly_values,
    _calculate_audio_revenue,
    _calculate_youtube_revenue,
    Constants,
)


def test_sanitize_numeric_input_bounds():
    assert sanitize_numeric_input(float('nan'), 0, 10) == 0
    # Non-finite values are coerced to min for safety
    assert sanitize_numeric_input(float('inf'), 0, 10) == 0
    assert sanitize_numeric_input(-5, 0, 10) == 0
    assert sanitize_numeric_input(15, 0, 10) == 10


def test_month0_values_default():
    audio = AudioInputs()
    youtube = YouTubeInputs()
    downloads, views = _calculate_monthly_values(audio, youtube, 0)
    assert downloads == audio.monthly_downloads
    assert views == youtube.monthly_views


def test_audio_and_youtube_revenue_positive():
    audio = AudioInputs()
    youtube = YouTubeInputs()
    downloads, views = _calculate_monthly_values(audio, youtube, 0)
    assert _calculate_audio_revenue(downloads, audio) > 0
    assert _calculate_youtube_revenue(views, youtube) > 0


def test_run_annual_projection_shape_and_nonnegative():
    inputs = {
        '_audio': AudioInputs(monthly_downloads=50_000).model_dump(),
        '_youtube': YouTubeInputs(monthly_views=100_000).model_dump(),
        '_other': OtherRevenue(subscriptions_monthly=1000, affiliate_monthly=500, other_monthly=200).model_dump(),
        '_costs': Costs(fixed_monthly=2000, variable_pct_gross=0.15).model_dump(),
        '_splits': Splits().model_dump(),
    }
    df = run_annual_projection(**inputs)
    assert df.shape[0] == Constants.MONTHS_IN_YEAR
    assert (df['Creator Net Revenue'] >= 0).all()
    assert df['Total Gross Revenue'].sum() > 0
