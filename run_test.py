import json

from core import AudioInputs, YouTubeInputs, OtherRevenue, Costs, Splits, run_annual_projection

state_inputs = {
    'audio': AudioInputs(),
    'youtube': YouTubeInputs(),
    'other': OtherRevenue(),
    'costs': Costs(),
    'splits': Splits(),
}
serializable = {
    '_audio': state_inputs['audio'].model_dump(),
    '_youtube': state_inputs['youtube'].model_dump(),
    '_other': state_inputs['other'].model_dump(),
    '_costs': state_inputs['costs'].model_dump(),
    '_splits': state_inputs['splits'].model_dump(),
}

df = run_annual_projection(**serializable)
annual_summary = df.sum(numeric_only=True)

result = {
    'rows': int(df.shape[0]),
    'cols': int(df.shape[1]),
    'total_gross_revenue': float(annual_summary['Total Gross Revenue']),
    'creator_net_revenue': float(annual_summary['Creator Net Revenue']),
}
print(json.dumps(result, indent=2))
