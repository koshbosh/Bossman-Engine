import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from regime_attribution import REGIMES, filter_by_regime, FACTORS_TO_TEST

def analyze():
    df = pd.read_csv("factor_raw_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    for regime_name, date_ranges in REGIMES.items():
        print(f"\\n{'='*50}\\nRegime: {regime_name}\\n{'='*50}")
        regime_df = filter_by_regime(df, date_ranges)
        
        for factor in FACTORS_TO_TEST:
            clean_df = regime_df[['forward_return', factor]].dropna().copy()
            if clean_df.empty: continue
            
            clean_df['quintile'] = pd.qcut(clean_df[factor], 5, labels=['Q5_Lowest', 'Q4', 'Q3', 'Q2', 'Q1_Highest'], duplicates='drop')
            q_rets = clean_df.groupby('quintile', observed=False)['forward_return'].mean() * 12
            
            q1 = q_rets.get('Q1_Highest', np.nan) * 100
            q2 = q_rets.get('Q2', np.nan) * 100
            q3 = q_rets.get('Q3', np.nan) * 100
            q4 = q_rets.get('Q4', np.nan) * 100
            q5 = q_rets.get('Q5_Lowest', np.nan) * 100
            
            # Check monotonicity
            monotonicity = ""
            if q1 > q2 and q2 > q3 and q3 > q4 and q4 > q5:
                monotonicity = "Strictly Positive Monotonic"
            elif q1 < q2 and q2 < q3 and q3 < q4 and q4 < q5:
                monotonicity = "Strictly Negative Monotonic"
            elif q1 >= q3 and q3 >= q5:
                monotonicity = "Generally Positive"
            elif q1 <= q3 and q3 <= q5:
                monotonicity = "Generally Negative"
            else:
                monotonicity = "Mixed / U-shaped"
                
            print(f"Factor: {factor: <15} | Q1: {q1:6.2f}% | Q2: {q2:6.2f}% | Q3: {q3:6.2f}% | Q4: {q4:6.2f}% | Q5: {q5:6.2f}% | Trend: {monotonicity}")

if __name__ == "__main__":
    analyze()
