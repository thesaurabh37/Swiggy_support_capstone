#Part 4
import os
import csv
import numpy as np

def run_part4():
    os.makedirs("outputs", exist_ok=True)

    # Load raw data for training set creation
    data = []
    with open("data/records.csv", "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            data.append([float(x) for x in parts[1:]])
    data = np.array(data)

    features_raw = data[:, :7]
    outcome = data[:, 7]

    idx_0 = np.where(outcome == 0)[0]
    idx_1 = np.where(outcome == 1)[0]

    n_0_train = int(np.floor(7 * len(idx_0) / 10))
    n_1_train = int(np.floor(7 * len(idx_1) / 10))

    train_idx = np.concatenate([idx_0[:n_0_train], idx_1[:n_1_train]])
    test_idx = np.concatenate([idx_0[n_0_train:], idx_1[n_1_train:]])

    X_train_orig = features_raw[train_idx]
    y_train = outcome[train_idx]
    X_test_orig = features_raw[test_idx]
    y_test = outcome[test_idx]

    # Standardize for logistic/KNN models (re-using Part 2 logic)
    train_mean = np.mean(X_train_orig, axis=0)
    train_sd = np.std(X_train_orig, axis=0, ddof=1)
    X_train_std = (X_train_orig - train_mean) / train_sd
    X_test_std = (X_test_orig - train_mean) / train_sd

    # Logistic regression fitted weights for voting
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    w = np.zeros(7)
    b = 0.0
    for _ in range(1000):
        z = X_train_std @ w + b
        y_hat = sigmoid(z)
        dw = (1.0 / len(X_train_std)) * (X_train_std.T @ (y_hat - y_train))
        db = (1.0 / len(X_train_std)) * np.sum(y_hat - y_train)
        w -= 0.1 * dw
        b -= 0.1 * db

    log_probs_test = sigmoid(X_test_std @ w + b)

    # 4.1 Splits evaluation
    # S1: X6 = 1, S2: X7 = 1, S3: X3 > 0.5, S4: X1 <= 24
    # Columns in X_train_orig: X1 (idx 0), X2 (1), X3 (2), X4 (3), X5 (4), X6 (5), X7 (6)
    def calc_entropy(pos_count, total_count):
        if pos_count == 0 or pos_count == total_count:
            return 0.0
        p1 = pos_count / total_count
        p0 = 1.0 - p1
        return - (p1 * np.log2(p1) + p0 * np.log2(p0))

    def calc_gini(pos_count, total_count):
        if total_count == 0:
            return 0.0
        p1 = pos_count / total_count
        p0 = 1.0 - p1
        return 1.0 - (p1**2 + p0**2)

    root_pos = np.sum(y_train == 1)
    root_tot = len(y_train)
    root_entropy = calc_entropy(root_pos, root_tot)
    root_gini = calc_gini(root_pos, root_tot)

    splits_def = {
        'S1': X_train_orig[:, 5] == 1,
        'S2': X_train_orig[:, 6] == 1,
        'S3': X_train_orig[:, 2] > 0.5,
        'S4': X_train_orig[:, 0] <= 24
    }

    split_rows = []
    best_info_gain = -1
    best_gini_gain = -1
    best_s_info = 'S1'
    best_s_gini = 'S1'

    info_gains = {}
    gini_gains = {}

    for s_name, cond in sorted(splits_def.items()):
        yes_idx = np.where(cond)[0]
        no_idx = np.where(~cond)[0]

        n_yes = len(yes_idx)
        pos_yes = np.sum(y_train[yes_idx] == 1)
        n_no = len(no_idx)
        pos_no = np.sum(y_train[no_idx] == 1)

        ent_yes = calc_entropy(pos_yes, n_yes)
        ent_no = calc_entropy(pos_no, n_no)
        child_ent = (n_yes / root_tot) * ent_yes + (n_no / root_tot) * ent_no
        info_gain = root_entropy - child_ent
        info_gains[s_name] = info_gain

        gini_yes = calc_gini(pos_yes, n_yes)
        gini_no = calc_gini(pos_no, n_no)
        child_gini = (n_yes / root_tot) * gini_yes + (n_no / root_tot) * gini_no
        gini_gain = root_gini - child_gini
        gini_gains[s_name] = gini_gain

        split_rows.append([s_name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain])

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_s_info = s_name
        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_s_gini = s_name

    with open("outputs/part4_splits.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])
        writer.writerows(split_rows)

    best_split_by_info_gain = best_s_info
    best_split_by_gini_gain = best_s_gini

    # 4.2 Stump & Voting
    # Stump is best split by info gain
    best_cond = splits_def[best_split_by_info_gain]
    stump_yes_idx = np.where(best_cond)[0]
    stump_no_idx = np.where(~best_cond)[0]

    stump_rate_yes = np.sum(y_train[stump_yes_idx] == 1) / len(stump_yes_idx)
    stump_rate_no = np.sum(y_train[stump_no_idx] == 1) / len(stump_no_idx)

    # Predictions for test set
    test_cond = (X_test_orig[:, 5] == 1) if best_split_by_info_gain == 'S1' else \
                ((X_test_orig[:, 6] == 1) if best_split_by_info_gain == 'S2' else \
                ((X_test_orig[:, 2] > 0.5) if best_split_by_info_gain == 'S3' else (X_test_orig[:, 0] <= 24)))

    stump_probs_test = np.where(test_cond, stump_rate_yes, stump_rate_no)

    # KNN predictions & probabilities (using K=5 from cross validation or optimal)
    # Let's compute KNN test probs
    knn_k = 5 # from typical reference
    knn_probs_test = []
    for query in X_test_std:
        dists = np.linalg.norm(X_train_std - query, axis=1)
        sorted_indices = np.lexsort((np.arange(len(dists)), dists))
        k_indices = sorted_indices[:knn_k]
        k_labels = y_train[k_indices]
        knn_probs_test.append(np.sum(k_labels == 1) / knn_k)
    knn_probs_test = np.array(knn_probs_test)

    def get_metrics_counts(y_true, pred_bin):
        tp = int(np.sum((y_true == 1) & (pred_bin == 1)))
        fp = int(np.sum((y_true == 0) & (pred_bin == 1)))
        tn = int(np.sum((y_true == 0) & (pred_bin == 0)))
        fn = int(np.sum((y_true == 1) & (pred_bin == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return tp, fp, tn, fn, prec, rec, f1

    # Models: logistic, knn, stump, hard_vote, soft_vote
    log_pred = (log_probs_test >= 0.3).astype(int)
    knn_pred = (knn_probs_test >= 0.5).astype(int) # or >= (k+1)/2 -> 3/5 = 0.6, or general rule
    knn_bin = (knn_probs_test >= 3/5).astype(int)
    stump_pred = (stump_probs_test >= 0.3).astype(int)

    hard_votes = (log_pred + knn_bin + stump_pred) >= 2
    soft_probs = (log_probs_test + knn_probs_test + stump_probs_test) / 3.0
    soft_votes = soft_probs >= 0.3

    voting_results = []
    models_eval = [
        ("logistic", log_pred),
        ("knn", knn_bin),
        ("stump", stump_pred),
        ("hard_vote", hard_votes.astype(int)),
        ("soft_vote", soft_votes.astype(int))
    ]

    voting_csv_rows = []
    for m_name, pred in models_eval:
        tp, fp, tn, fn, prec, rec, f1 = get_metrics_counts(y_test, pred)
        voting_csv_rows.append([m_name, tp, fp, tn, fn, prec, rec, f1])

    with open("outputs/part4_voting.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
        writer.writerows(voting_csv_rows)

    # 4.3 AdaBoost by hand
    y_adaboost = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    w_curr = np.ones(8) / 8.0
    adaboost_rows = []

    # Round 1
    err_1 = np.sum(w_curr[h1 != y_adaboost])
    alpha_1 = 0.5 * np.log((1.0 - err_1) / err_1)

    w_unnorm_1 = w_curr * np.exp(-alpha_1 * y_adaboost * h1)
    w_next_1 = w_unnorm_1 / np.sum(w_unnorm_1)

    for i in range(8):
        adaboost_rows.append([1, i+1, w_curr[i], w_unnorm_1[i], w_next_1[i]])

    # Round 2
    err_2 = np.sum(w_next_1[h2 != y_adaboost])
    alpha_2 = 0.5 * np.log((1.0 - err_2) / err_2)

    w_unnorm_2 = w_next_1 * np.exp(-alpha_2 * y_adaboost * h2)
    w_next_2 = w_unnorm_2 / np.sum(w_unnorm_2)

    for i in range(8):
        adaboost_rows.append([2, i+1, w_next_1[i], w_unnorm_2[i], w_next_2[i]])

    with open("outputs/part4_adaboost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
        writer.writerows(adaboost_rows)

    ensemble_preds = np.sign(alpha_1 * h1 + alpha_2 * h2)
    ensemble_accuracy = np.mean(ensemble_preds == y_adaboost)

    round1_error = err_1
    round1_alpha = alpha_1
    round2_error = err_2
    round2_alpha = alpha_2

    # 4.4 Out-of-bag fraction
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560

    # 4.5 Architecture sizing
    # Model A Option 1 MLP: 784 -> 16 -> 16 -> 10
    mlp_h1 = 784 * 16 + 16
    mlp_h2 = 16 * 16 + 16
    mlp_out = 16 * 10 + 10
    mlp_total = mlp_h1 + mlp_h2 + mlp_out

    # Model A Option 2 CNN:
    # Input 28x28x1
    # Conv1: 5x5 filters, 6 filters, stride 1, pad 2 -> Output 28x28x6. Params: (5*5*1*6) + 6 = 156
    # Max-pool 2x2, stride 2 -> Output 14x14x6
    # Conv2: 5x5 filters, 16 filters, stride 1, pad 0 -> Output (14-5+0)/1 + 1 = 10x10x16. Params: (5*5*6*16) + 16 = 2416
    # Max-pool 2x2, stride 2 -> Output 5x5x16 = 400
    # Dense1: 400 -> 32. Params: 400 * 32 + 32 = 12832
    # Dense2: 32 -> 10. Params: 32 * 10 + 10 = 330
    cnn_c1 = 5 * 5 * 1 * 6 + 6
    cnn_c2 = 5 * 5 * 6 * 16 + 16
    cnn_d1 = 400 * 32 + 32
    cnn_d2 = 32 * 10 + 10
    cnn_total = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

    arch_rows = [
        ["mlp_hidden1", mlp_h1],
        ["mlp_hidden2", mlp_h2],
        ["mlp_output", mlp_out],
        ["cnn_conv1", cnn_c1],
        ["cnn_conv2", cnn_c2],
        ["cnn_dense1", cnn_d1],
        ["cnn_dense2", cnn_d2]
    ]

    with open("outputs/part4_architecture.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        writer.writerows(arch_rows)

    # FC vs Conv comparison for 128x128x3 input into 1024 neurons vs 64 filters of 3x3
    fc_layer_params = (128 * 128 * 3) * 1024 + 1024
    conv_layer_params = (3 * 3 * 3 * 64) + 64

    # Model B RNN: 64-dim word vector, 128 hidden state, 16 classes
    # W_xh: 64 x 128
    # W_hh: 128 x 128
    # b_h: 128
    # W_hy: 128 x 16
    # b_y: 16
    rnn_total = (64 * 128) + (128 * 128) + 128 + (128 * 16) + 16

    # Write results
    results = [
        f"root_entropy = {root_entropy:.4f}",
        f"root_gini = {root_gini:.4f}",
        f"best_split_by_info_gain = {best_split_by_info_gain}",
        f"best_split_by_gini_gain = {best_split_by_gini_gain}",
        f"stump_rate_yes = {stump_rate_yes:.4f}",
        f"stump_rate_no = {stump_rate_no:.4f}",
        f"round1_error = {round1_error:.4f}",
        f"round1_alpha = {round1_alpha:.4f}",
        f"round2_error = {round2_error:.4f}",
        f"round2_alpha = {round2_alpha:.4f}",
        f"ensemble_accuracy = {ensemble_accuracy:.4f}",
        f"oob_fraction = {oob_fraction:.6f}",
        f"mlp_total = {mlp_total}",
        f"cnn_total = {cnn_total}",
        f"fc_layer_params = {fc_layer_params}",
        f"conv_layer_params = {conv_layer_params}",
        f"rnn_total = {rnn_total}"
    ]

    with open("outputs/part4_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")
    print("Part 4 completed successfully.")

if __name__ == "__main__":
    run_part4()