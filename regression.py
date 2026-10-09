import numpy as np
import pandas as pd
import math
import json
import os
from dateutil.relativedelta import relativedelta
from recommender import calculate_ma_proximity, SECTOR_PE_BENCHMARKS

class NumPyOLS:
    """
    An Ordinary Least Squares (OLS) regression model implemented in pure NumPy.
    Supports standard scaling, prediction interval computation, and serialization.
    """
    def __init__(self):
        self.beta = None
        self.se = None
        self.t_stats = None
        self.p_values = None
        self.r2 = None
        self.adj_r2 = None
        self.sigma = None
        self.xtx_inv = None
        self.feature_names = None
        self.means = None
        self.stds = None
        self.n_obs = None

    def fit(self, X_raw, y, feature_names):
        """
        Fits OLS regression: y = beta_0 + beta_1 * X_1 + ... + beta_p * X_p + e.
        Scale features using standard scaling.
        """
        self.feature_names = list(feature_names)
        self.n_obs = len(y)
        
        # Scale features
        self.means = np.mean(X_raw, axis=0)
        self.stds = np.std(X_raw, axis=0)
        # Avoid division by zero
        self.stds = np.where(self.stds == 0, 1.0, self.stds)
        X_scaled = (X_raw - self.means) / self.stds
        
        N, P_minus_1 = X_scaled.shape
        X = np.hstack([np.ones((N, 1)), X_scaled])  # Add constant (intercept)
        P = P_minus_1 + 1
        
        # OLS estimator: beta = (X^T X)^-1 X^T y
        try:
            xtx = X.T @ X
            self.xtx_inv = np.linalg.inv(xtx)
            self.beta = self.xtx_inv @ X.T @ y
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in OLS. Using pseudo-inverse.")
            self.xtx_inv = np.linalg.pinv(xtx)
            self.beta = self.xtx_inv @ X.T @ y
            
        y_pred = X @ self.beta
        residuals = y - y_pred
        
        # Diagnostics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))
        self.adj_r2 = float(1.0 - (1.0 - self.r2) * (N - 1) / (N - P))
        
        # Residual standard error
        self.sigma = float(np.sqrt(ss_res / (N - P)))
        
        # Standard errors of coefficients
        self.se = np.sqrt(np.diagonal(self.xtx_inv) * (self.sigma ** 2))
        self.t_stats = self.beta / (self.se + 1e-9)
        
        # 2-tailed p-values using normal approximation (since N is large)
        self.p_values = []
        for t in self.t_stats:
            p = 2.0 * (1.0 - self.normal_cdf(abs(t)))
            self.p_values.append(p)
        self.p_values = np.array(self.p_values)

    def predict(self, X_raw):
        """
        Predicts expected returns and prediction standard errors for new observations.
        Returns:
            y_pred: numpy array of predicted values.
            se_pred: numpy array of prediction standard errors (includes residual variance).
        """
        if self.beta is None:
            raise ValueError("Model is not fitted yet!")
            
        X_scaled = (X_raw - self.means) / self.stds
        N = X_scaled.shape[0]
        X = np.hstack([np.ones((N, 1)), X_scaled])
        
        # Point prediction
        y_pred = X @ self.beta
        
        # Prediction standard error (for individual stock returns):
        # se_pred_i = sigma * sqrt(1 + x_i^T (X^T X)^-1 x_i)
        se_pred = []
        for i in range(N):
            x_i = X[i]
            se = self.sigma * np.sqrt(1.0 + x_i.T @ self.xtx_inv @ x_i)
            se_pred.append(se)
            
        return y_pred, np.array(se_pred)

    @staticmethod
    def normal_cdf(z):
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def save_model(self, file_path):
        """Saves fitted model parameters to a JSON file."""
        model_data = {
            "beta": self.beta.tolist(),
            "se": self.se.tolist(),
            "t_stats": self.t_stats.tolist(),
            "p_values": self.p_values.tolist(),
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "sigma": self.sigma,
            "xtx_inv": self.xtx_inv.tolist(),
            "feature_names": self.feature_names,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "n_obs": self.n_obs
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)
        print(f"Model successfully saved to {file_path}")

    def load_model(self, file_path):
        """Loads model parameters from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        with open(file_path, "r") as f:
            model_data = json.load(f)
            
        self.beta = np.array(model_data["beta"])
        self.se = np.array(model_data["se"])
        self.t_stats = np.array(model_data["t_stats"])
        self.p_values = np.array(model_data["p_values"])
        self.r2 = model_data["r2"]
        self.adj_r2 = model_data["adj_r2"]
        self.sigma = model_data["sigma"]
        self.xtx_inv = np.array(model_data["xtx_inv"])
        self.feature_names = model_data["feature_names"]
        self.means = np.array(model_data["means"])
        self.stds = np.array(model_data["stds"])
        self.n_obs = model_data["n_obs"]
        print(f"Model successfully loaded from {file_path}")

    def summary(self):
        """Returns a string representation of the OLS summary table."""
        if self.beta is None:
            return "Model not fitted."
            
        lines = []
        lines.append("="*85)
        lines.append("                 BASAK BOSSMAN ENGINE: OLS REGRESSION SUMMARY")
        lines.append("="*85)
        lines.append(f"Observations (N):   {self.n_obs:<10} | R-squared:          {self.r2:.5f}")
        lines.append(f"Residual Std Error: {self.sigma:<10.5f} | Adj. R-squared:     {self.adj_r2:.5f}")
        lines.append("-"*85)
        lines.append(f"{'Feature Factor':<25} | {'Coef':>10} | {'Std Err':>10} | {'t-stat':>10} | {'p-value':>12}")
        lines.append("-"*85)
        
        # Intercept (constant)
        lines.append(f"{'Intercept (Constant)':<25} | {self.beta[0]:>10.4f} | {self.se[0]:>10.4f} | {self.t_stats[0]:>10.2f} | {self.p_values[0]:>12.4e}")
        
        # Features
        for name, coef, se, t, p in zip(self.feature_names, self.beta[1:], self.se[1:], self.t_stats[1:], self.p_values[1:]):
            lines.append(f"{name:<25} | {coef:>10.4f} | {se:>10.4f} | {t:>10.2f} | {p:>12.4e}")
        lines.append("="*85)
        return "\n".join(lines)

def calculate_beta(stock_returns, market_returns):
    if len(stock_returns) < 60 or len(market_returns) < 60:
        return 1.0
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var != 0 else 1.0

def get_trained_ols_model(tickers, price_data, fund_data, sector_map, model_file="regression_model.json"):
    """
    Attempts to load a pre-trained regression model from JSON.
    Falls back to dynamic training on local 2-year scan data if missing.
    """
    ols = NumPyOLS()
    if os.path.exists(model_file):
        try:
            ols.load_model(model_file)
            print(f"\nLoaded 25-Year OLS regression model weights from '{model_file}':")
            print(ols.summary())
            return ols
        except Exception as e:
            print(f"Warning: Failed to load {model_file} ({e}). Re-training dynamically...")
    else:
        print(f"\nModel file '{model_file}' not found. Training dynamically on the local price history...")

    # Dynamic training fallback
    monthly_prices = price_data.resample('ME').last()
    spy_rets = price_data["SPY"].pct_change().dropna()
    
    # We need at least 1 year (252 days) of daily history for features,
    # and 1 month of forward history for target label.
    # So we sample dates in the range of [1 year ago, 1 month ago]
    available_dates = [d for d in monthly_prices.index if d > monthly_prices.index[0] + relativedelta(years=1) and d < monthly_prices.index[-1]]
    
    training_rows = []
    features_list = ["pe_ratio", "debt_to_equity", "ma50_proximity", "ma200_proximity", "momentum_1y", "rsi", "annualized_vol", "beta"]
    
    print(f"Extracting features across {len(available_dates)} historical dates...")
    for date in available_dates:
        next_date = monthly_prices.index[monthly_prices.index.get_loc(date) + 1]
        hist_p = price_data.loc[:date]
        market_rets = spy_rets.loc[:date].iloc[-252:]
        
        for t in tickers:
            if t not in hist_p.columns or t == "SPY":
                continue
            s = hist_p[t].dropna()
            if len(s) < 252:
                continue
            
            p_now = s.iloc[-1]
            p_next = monthly_prices.loc[next_date, t]
            if pd.isna(p_now) or pd.isna(p_next) or p_now <= 0 or p_next <= 0:
                continue
            fwd_return = (p_next / p_now) - 1.0
            
            s_rets = s.iloc[-252:].pct_change().dropna()
            m1y = (p_now / s.iloc[-252] - 1)
            
            # RSI
            delta = s.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            
            vol = s_rets.std() * np.sqrt(252)
            beta = calculate_beta(s_rets, market_rets)
            
            ma_50 = float(s.iloc[-50:].mean()) if len(s) >= 50 else float("nan")
            ma_200 = float(s.iloc[-200:].mean()) if len(s) >= 200 else float("nan")
            if np.isnan(ma_50) or np.isnan(ma_200):
                continue
            
            ma50_prox = calculate_ma_proximity(p_now, ma_50)
            ma200_prox = calculate_ma_proximity(p_now, ma_200)
            
            funds = fund_data.get(t, {})
            current_pe = funds.get("pe_ratio")
            de = funds.get("debt_to_equity", 0)
            
            price_current = price_data[t].iloc[-1]
            if current_pe is not None and price_current > 0 and p_now > 0:
                sim_pe = current_pe * (p_now / price_current)
            else:
                sector = sector_map.get(t, "N/A")
                sim_pe = SECTOR_PE_BENCHMARKS.get(sector, 23.5)
            
            if sim_pe <= 0 or np.isnan(sim_pe):
                sector = sector_map.get(t, "N/A")
                sim_pe = SECTOR_PE_BENCHMARKS.get(sector, 23.5)
                
            sim_pe = np.clip(sim_pe, 2.0, 150.0)
            de = np.clip(de, 0.0, 500.0)
            
            training_rows.append({
                "forward_return": fwd_return,
                "pe_ratio": sim_pe,
                "debt_to_equity": de,
                "ma50_proximity": ma50_prox,
                "ma200_proximity": ma200_prox,
                "momentum_1y": m1y,
                "rsi": rsi,
                "annualized_vol": vol,
                "beta": beta
            })
            
    df_train = pd.DataFrame(training_rows)
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna()
    df_train = df_train[(df_train['forward_return'] >= -0.50) & (df_train['forward_return'] <= 0.50)]
    
    print(f"Extracted {len(df_train)} observations. Fitting OLS...")
    X = df_train[features_list].values
    y = df_train["forward_return"].values
    
    ols.fit(X, y, features_list)
    print(ols.summary())
    return ols
