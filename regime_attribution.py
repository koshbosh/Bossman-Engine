import pandas as pd
import numpy as np
import os
import warnings

from regression import NumPyOLS

warnings.simplefilter(action='ignore', category=FutureWarning)

FACTORS_TO_TEST = [
    'momentum_1mo', 'momentum_3mo', 'momentum_6mo', 'momentum_1y',
    'ma50_proximity', 'ma200_proximity', 'rsi', 'annualized_vol'
]

REGIMES = {
    'Bull Market (Alpha Bull)': [
        ('2017-01-01', '2018-01-31'),
        ('2019-01-01', '2020-01-31'),
        ('2020-04-01', '2021-12-31'),
        ('2023-01-01', '2026-06-30')
    ],
    'Bear Market (Bear Shield / Fundamental Bag)': [
        ('2018-02-01', '2018-12-31'),
        ('2020-02-01', '2020-03-31'),
        ('2022-01-01', '2022-10-31')
    ],
    'Sideways / Choppy (Dip Hunter)': [
        ('2016-01-01', '2016-12-31'),
        ('2022-11-01', '2022-12-31')
    ],
    'Full Cycle (All-Weather Core)': [
        ('2016-01-01', '2026-12-31')
    ]
}

def filter_by_regime(df, date_ranges):
    # df['date'] is already datetime
    mask = pd.Series([False] * len(df), index=df.index)
    for start, end in date_ranges:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        mask = mask | ((df['date'] >= s) & (df['date'] <= e))
    return df[mask]

def run_regime_analysis(results_df):
    results_df['date'] = pd.to_datetime(results_df['date'])
    
    all_regime_stats = {}
    
    for regime_name, date_ranges in REGIMES.items():
        print(f"\n--- Analyzing Regime: {regime_name} ---")
        regime_df = filter_by_regime(results_df, date_ranges)
        print(f"Observations: {len(regime_df):,}")
        
        if regime_df.empty:
            continue
            
        clean_df = regime_df[['forward_return'] + FACTORS_TO_TEST].dropna().copy()
        
        # 1. Multivariate OLS
        X = clean_df[FACTORS_TO_TEST].values
        y = clean_df['forward_return'].values
        
        ols = NumPyOLS()
        ols.fit(X, y, FACTORS_TO_TEST)
        
        factor_stats = {}
        for i, factor in enumerate(FACTORS_TO_TEST):
            coef = float(ols.beta[i+1])
            t_stat = float(ols.t_stats[i+1])
            p_val = float(ols.p_values[i+1])
            
            # 2. Single Factor Spearman Correlation (Pearson on ranks)
            spearman_corr = clean_df[factor].rank().corr(clean_df['forward_return'].rank(), method='pearson')
            
            # 3. Quintiles
            clean_df['quintile'] = pd.qcut(clean_df[factor], 5, labels=['Q5_Lowest', 'Q4', 'Q3', 'Q2', 'Q1_Highest'], duplicates='drop')
            q_rets = clean_df.groupby('quintile', observed=False)['forward_return'].mean() * 12
            
            q1_ret = q_rets.get('Q1_Highest', float('nan'))
            q5_ret = q_rets.get('Q5_Lowest', float('nan'))
            
            factor_stats[factor] = {
                'multi_coef': coef,
                'multi_t': t_stat,
                'multi_p': p_val,
                'single_corr': float(spearman_corr),
                'q1_ret': float(q1_ret),
                'q5_ret': float(q5_ret),
                'spread': float(q1_ret - q5_ret)
            }
            
        all_regime_stats[regime_name] = {
            'n_obs': len(clean_df),
            'stats': factor_stats
        }
        
    generate_html_report(all_regime_stats)


def generate_html_report(all_regime_stats):
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Regime-Based Factor Attribution</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #58a6ff; }}
            .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #30363d; padding: 12px; text-align: left; }}
            th {{ background-color: #21262d; font-weight: 600; }}
            .positive {{ color: #3fb950; }}
            .negative {{ color: #f85149; }}
            .signal {{ color: #a371f7; font-weight: bold; }}
            .noise {{ color: #8b949e; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Regime-Based Factor Attribution</h1>
            <p>Evaluating factors across specific market environments to align with profile objectives.</p>
    """
    
    for regime_name, regime_data in all_regime_stats.items():
        n_obs = regime_data['n_obs']
        stats = regime_data['stats']
        
        # Sort by absolute multivariate t-stat descending
        sorted_factors = sorted(stats.items(), key=lambda x: abs(x[1]['multi_t']), reverse=True)
        
        html += f"""
            <div class="card">
                <h2>{regime_name}</h2>
                <p><strong>N = {n_obs:,} observations</strong></p>
                <table>
                    <tr>
                        <th>Factor</th>
                        <th>Multivariate t-stat</th>
                        <th>Multivariate p-val</th>
                        <th>Single-Factor Corr</th>
                        <th>Q1 (Highest) Return</th>
                        <th>Q5 (Lowest) Return</th>
                        <th>Independent Signal?</th>
                    </tr>
        """
        
        for factor, fstats in sorted_factors:
            is_significant = fstats['multi_p'] < 0.05
            verdict = "<span class='signal'>YES</span>" if is_significant else "<span class='noise'>NO</span>"
            
            html += f"""
                    <tr>
                        <td><strong>{factor}</strong></td>
                        <td class="{'positive' if fstats['multi_t'] > 0 else 'negative'}">{fstats['multi_t']:.2f}</td>
                        <td class="{'positive' if is_significant else ''}">{fstats['multi_p']:.4e}</td>
                        <td class="{'positive' if fstats['single_corr'] > 0 else 'negative'}">{fstats['single_corr']:.4f}</td>
                        <td class="{'positive' if fstats['q1_ret'] > 0 else 'negative'}">{(fstats['q1_ret']*100):.2f}%</td>
                        <td class="{'positive' if fstats['q5_ret'] > 0 else 'negative'}">{(fstats['q5_ret']*100):.2f}%</td>
                        <td>{verdict}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
    html += """
        </div>
    </body>
    </html>
    """
    
    with open("regime_attribution_results.html", "w") as f:
        f.write(html)
    print("Report saved to regime_attribution_results.html")

if __name__ == "__main__":
    if not os.path.exists("factor_raw_data.csv"):
        print("Error: factor_raw_data.csv not found. Please run factor_attribution.py first.")
    else:
        df = pd.read_csv("factor_raw_data.csv")
        run_regime_analysis(df)
