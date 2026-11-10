#Part 1
import os
import numpy as np

def run_part1():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    data = []
    with open("data/records.csv", "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            # columns: id, X1, X2, X3, X4, X5, X6, X7, outcome, value
            data.append([float(x) for x in parts[1:]])
    data = np.array(data)

    X1 = data[:, 0]
    X2 = data[:, 1]
    X3 = data[:, 2]
    X4 = data[:, 3]
    X5 = data[:, 4]
    X6 = data[:, 5]
    X7 = data[:, 6]
    outcome = data[:, 7]
    value = data[:, 8]

    n = len(value)

    # 1.1 Estimation
    mean_value = np.mean(value)
    sd_value = np.std(value, ddof=1)
    se_value = sd_value / np.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # 1.2 Test planning claim (mu = 8.0)
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # 1.3 Sample size for E = 0.5
    E = 0.5
    n_required = int(np.ceil((1.96 * sd_value / E) ** 2))

    # 1.4 Simple regression of value on X4
    cov_x4_val = np.cov(X4, value, ddof=1)[0, 1]
    var_x4 = np.var(X4, ddof=1)
    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_value - reg_slope * np.mean(X4)
    fitted_14 = reg_slope * X4 + reg_intercept

    tss = np.sum((value - mean_value) ** 2)
    ess = np.sum((fitted_14 - mean_value) ** 2)
    rss = np.sum((value - fitted_14) ** 2)
    tss_equals_ess_plus_rss = "yes" if abs(tss - (ess + rss)) < 10**-6 * tss else "no"
    r2_14 = 1.0 - (rss / tss)
    rmse_14 = np.sqrt(rss / n)
    mae_14 = np.mean(np.abs(value - fitted_14))

    # 1.5 Gradient descent on standardized X4
    mean_x4 = np.mean(X4)
    sd_x4 = np.std(X4, ddof=1)
    z_x4 = (X4 - mean_x4) / sd_x4

    m = 0.0
    c = 0.0
    eta = 0.1

    def compute_cost(m_val, c_val):
        y_hat = m_val * z_x4 + c_val
        e = value - y_hat
        return np.mean(e ** 2)

    gd_j0 = compute_cost(m, c)
    costs = [gd_j0]

    for step in range(1, 501):
        y_hat = m * z_x4 + c
        e = value - y_hat
        dm = -(2.0 / n) * np.sum(z_x4 * e)
        dc = -(2.0 / n) * np.sum(e)
        m -= eta * dm
        c -= eta * dc
        j_curr = np.mean(e ** 2)
        if step == 1: gd_j1 = j_curr
        if step == 10: gd_j10 = j_curr
        if step == 100: gd_j100 = j_curr
        if step == 500: gd_j500 = j_curr

    gd_slope = m / sd_x4
    gd_intercept = c - m * mean_x4 / sd_x4
    gd_slope_gap = gd_slope - reg_slope

    # 1.6 Funnel check & Log-log model
    sort_idx = np.argsort(fitted_14, kind='stable')
    sorted_fitted = fitted_14[sort_idx]
    sorted_resids = (value - fitted_14)[sort_idx]

    lower_resids = sorted_resids[:400]
    upper_resids = sorted_resids[400:]
    resid_sd_lower_raw = np.std(lower_resids, ddof=1)
    resid_sd_upper_raw = np.std(upper_resids, ddof=1)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log model: ln(value) on ln(X4)
    ln_val = np.log(value)
    ln_x4 = np.log(X4)
    mean_ln_x4 = np.mean(ln_x4)
    mean_ln_val = np.mean(ln_val)
    cov_ln = np.cov(ln_x4, ln_val, ddof=1)[0, 1]
    var_ln_x4 = np.var(ln_x4, ddof=1)
    loglog_slope = cov_ln / var_ln_x4
    loglog_intercept = mean_ln_val - loglog_slope * mean_ln_x4
    fitted_loglog = loglog_slope * ln_x4 + loglog_intercept

    loglog_rss = np.sum((ln_val - fitted_loglog) ** 2)
    loglog_tss = np.sum((ln_val - mean_ln_val) ** 2)
    loglog_r2 = 1.0 - (loglog_rss / loglog_tss)
    loglog_rmse = np.sqrt(loglog_rss / n)

    sort_idx_log = np.argsort(fitted_loglog, kind='stable')
    sorted_log_resids = (ln_val - fitted_loglog)[sort_idx_log]
    resid_sd_lower_log = np.std(sorted_log_resids[:400], ddof=1)
    resid_sd_upper_log = np.std(sorted_log_resids[400:], ddof=1)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # 1.7 Multiple regression via normal equation
    # X columns: constant 1, ln(X4), X1, X2, X3, X5, X6, X7
    X_mat = np.column_stack([np.ones(n), ln_x4, X1, X2, X3, X5, X6, X7])
    Y_mat = ln_val

    # beta = (X^T X)^(-1) X^T Y
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ Y_mat)
    mr_beta_0, mr_beta_1, mr_beta_2, mr_beta_3, mr_beta_4, mr_beta_5, mr_beta_6, mr_beta_7 = beta

    mr_fitted = X_mat @ beta
    mr_rss = np.sum((Y_mat - mr_fitted) ** 2)
    mr_tss = np.sum((Y_mat - mean_ln_val) ** 2)
    mr_r2 = 1.0 - (mr_rss / mr_tss)
    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    # Write results
    results = [
        f"n = {n}",
        f"mean_value = {mean_value:.4f}",
        f"sd_value = {sd_value:.4f}",
        f"se_value = {se_value:.4f}",
        f"ci95_low = {ci95_low:.4f}",
        f"ci95_high = {ci95_high:.4f}",
        f"z_stat = {z_stat:.4f}",
        f"reject_h0 = {reject_h0}",
        f"n_required = {n_required}",
        f"reg_slope = {reg_slope:.4f}",
        f"reg_intercept = {reg_intercept:.4f}",
        f"tss_equals_ess_plus_rss = {tss_equals_ess_plus_rss}",
        f"r2 = {r2_14:.4f}",
        f"rmse = {rmse_14:.4f}",
        f"mae = {mae_14:.4f}",
        f"gd_j0 = {gd_j0:.4f}",
        f"gd_j1 = {gd_j1:.4f}",
        f"gd_j10 = {gd_j10:.4f}",
        f"gd_j100 = {gd_j100:.4f}",
        f"gd_j500 = {gd_j500:.4f}",
        f"gd_slope = {gd_slope:.4f}",
        f"gd_intercept = {gd_intercept:.4f}",
        f"gd_slope_gap = {gd_slope_gap:.4f}",
        f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}",
        f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}",
        f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}",
        f"loglog_intercept = {loglog_intercept:.4f}",
        f"loglog_slope = {loglog_slope:.4f}",
        f"loglog_r2 = {loglog_r2:.4f}",
        f"loglog_rmse = {loglog_rmse:.4f}",
        f"resid_sd_lower_log = {resid_sd_lower_log:.4f}",
        f"resid_sd_upper_log = {resid_sd_upper_log:.4f}",
        f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}",
        f"mr_beta_0 = {mr_beta_0:.4f}",
        f"mr_beta_1 = {mr_beta_1:.4f}",
        f"mr_beta_2 = {mr_beta_2:.4f}",
        f"mr_beta_3 = {mr_beta_3:.4f}",
        f"mr_beta_4 = {mr_beta_4:.4f}",
        f"mr_beta_5 = {mr_beta_5:.4f}",
        f"mr_beta_6 = {mr_beta_6:.4f}",
        f"mr_beta_7 = {mr_beta_7:.4f}",
        f"mr_r2 = {mr_r2:.4f}",
        f"mr_adj_r2 = {mr_adj_r2:.4f}",
        f"loglog_adj_r2 = {loglog_adj_r2:.4f}"
    ]

    with open("outputs/part1_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")
    print("Part 1 completed successfully.")

if __name__ == "__main__":
    run_part1()