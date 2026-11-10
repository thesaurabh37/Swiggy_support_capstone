#Part 2
import os
import csv
import numpy as np

def run_part2():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    data = []
    with open("data/records.csv", "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            data.append([float(x) for x in parts[1:]])
    data = np.array(data)

    features = data[:, :7]
    outcome = data[:, 7]

    # 2.1 Stratified 70/30 split
    idx_0 = np.where(outcome == 0)[0]
    idx_1 = np.where(outcome == 1)[0]

    n_0_train = int(np.floor(7 * len(idx_0) / 10))
    n_1_train = int(np.floor(7 * len(idx_1) / 10))

    train_idx = np.concatenate([idx_0[:n_0_train], idx_1[:n_1_train]])
    test_idx = np.concatenate([idx_0[n_0_train:], idx_1[n_1_train:]])

    X_train_raw = features[train_idx]
    y_train = outcome[train_idx]

    X_test_raw = features[test_idx]
    y_test = outcome[test_idx]

    train_rows = len(X_train_raw)
    train_positives = int(np.sum(y_train))
    test_rows = len(X_test_raw)
    test_positives = int(np.sum(y_test))

    # Standardize based on train set
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    # 2.2 Logistic regression from scratch
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def train_logistic(X, y, eta=0.1, epochs=1000, return_losses_at=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        losses = {}

        for epoch in range(epochs + 1):
            z = X @ w + b
            y_hat = sigmoid(z)
            if return_losses_at and epoch in return_losses_at:
                # BCE loss
                eps = 1e-15
                y_hat_clip = np.clip(y_hat, eps, 1 - eps)
                loss = -np.mean(y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip))
                losses[epoch] = loss
            if epoch == epochs:
                break
            # Gradients
            dw = (1.0 / n_samples) * (X.T @ (y_hat - y))
            db = (1.0 / n_samples) * np.sum(y_hat - y)
            w -= eta * dw
            b -= eta * db
        return w, b, losses

    # Train with eta=0.1 for 1000 epochs tracking specific epochs
    track_epochs = [0, 1, 10, 100, 1000]
    w_fin, b_fin, losses_track = train_logistic(X_train, y_train, eta=0.1, epochs=1000, return_losses_at=track_epochs)

    loss_epoch_0 = losses_track[0]
    loss_epoch_1 = losses_track[1]
    loss_epoch_10 = losses_track[10]
    loss_epoch_100 = losses_track[100]
    loss_epoch_1000 = losses_track[1000]

    # Test BCE
    z_test = X_test @ w_fin + b_fin
    y_hat_test = sigmoid(z_test)
    eps = 1e-15
    y_hat_test_clip = np.clip(y_hat_test, eps, 1 - eps)
    test_bce = -np.mean(y_test * np.log(y_hat_test_clip) + (1 - y_test) * np.log(1 - y_hat_test_clip))

    # 2.3 Learning rate effect (200 epochs)
    _, _, l_001 = train_logistic(X_train, y_train, eta=0.01, epochs=200, return_losses_at=[200])
    _, _, l_01 = train_logistic(X_train, y_train, eta=0.1, epochs=200, return_losses_at=[200])
    _, _, l_1 = train_logistic(X_train, y_train, eta=1.0, epochs=200, return_losses_at=[200])

    loss_epoch_200_eta_001 = l_001[200]
    loss_epoch_200_eta_01 = l_01[200]
    loss_epoch_200_eta_10 = l_1[200]

    # 2.4 Odds ratios
    odds_ratios = np.exp(w_fin)

    # Helper for metrics calculation
    def get_metrics(y_true, y_pred_bin):
        tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))

        acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return tp, fp, tn, fn, acc, prec, rec, f1, spec

    # 2.5 Evaluate at tau = 0.5 and 0.3
    z_test_full = X_test @ w_fin + b_fin
    p_test = sigmoid(z_test_full)

    tp_05, fp_05, tn_05, fn_05, acc_05, prec_05, rec_05, f1_05, spec_05 = get_metrics(y_test, (p_test >= 0.5).astype(int))
    tp_03, fp_03, tn_03, fn_03, acc_03, prec_03, rec_03, f1_03, spec_03 = get_metrics(y_test, (p_test >= 0.3).astype(int))

    blind_accuracy = int(np.sum(y_test == 0)) / len(y_test)

    # 2.6 Threshold sweep for ROC
    thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
    threshold_rows = []
    for t in thresholds:
        pred_t = (p_test >= t).astype(int)
        tp, fp, tn, fn, _, _, rec, _, spec = get_metrics(y_test, pred_t)
        tpr = rec
        fpr = 1.0 - spec
        threshold_rows.append([t, tp, fp, tn, fn, tpr, fpr])

    with open("outputs/part2_thresholds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        writer.writerows(threshold_rows)

    # 2.7 KNN with 5-fold CV
    k_values = list(range(1, 22, 2))
    cv_rows = []
    best_k = 1
    best_mean_f1 = -1.0

    for k in k_values:
        fold_f1s = []
        for fold in range(5):
            val_idx = np.where(np.arange(train_rows) % 5 == fold)[0]
            ref_idx = np.where(np.arange(train_rows) % 5 != fold)[0]

            X_ref = X_train[ref_idx]
            y_ref = y_train[ref_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]

            # Predict KNN
            preds = []
            for query in X_val:
                dists = np.linalg.norm(X_ref - query, axis=1)
                # stable sort by distance, then file order
                sorted_indices = np.lexsort((np.arange(len(dists)), dists))
                k_indices = sorted_indices[:k]
                k_labels = y_ref[k_indices]
                pos_count = np.sum(k_labels == 1)
                pred = 1 if pos_count >= (k + 1) / 2 else 0
                preds.append(pred)
            _, _, _, _, _, _, _, f1_fold, _ = get_metrics(y_val, np.array(preds))
            fold_f1s.append(f1_fold)
        mean_cv_f1 = np.mean(fold_f1s)
        cv_rows.append([k, mean_cv_f1])
        if mean_cv_f1 > best_mean_f1:
            best_mean_f1 = mean_cv_f1
            best_k = k

    with open("outputs/part2_knn_cv.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        writer.writerows(cv_rows)

    knn_best_k = best_k

    # Refit KNN with all train rows
    knn_preds = []
    for query in X_test:
        dists = np.linalg.norm(X_train - query, axis=1)
        sorted_indices = np.lexsort((np.arange(len(dists)), dists))
        k_indices = sorted_indices[:knn_best_k]
        k_labels = y_train[k_indices]
        pos_count = np.sum(k_labels == 1)
        pred = 1 if pos_count >= (knn_best_k + 1) / 2 else 0
        knn_preds.append(pred)

    knn_tp, knn_fp, knn_tn, knn_fn, knn_acc, knn_prec, knn_rec, knn_f1, knn_spec = get_metrics(y_test, np.array(knn_preds))

    # 2.8 Money threshold optimization
    z_train_full = X_train @ w_fin + b_fin
    p_train = sigmoid(z_train_full)

    best_tau = 0.5
    min_train_cost = float("inf")

    for k_val in range(1, 20):
        tau = k_val / 20.0
        pred_train = (p_train >= tau).astype(int)
        tp, fp, tn, fn = get_metrics(y_train, pred_train)[:4]
        cost = 50 * fp + 400 * fn
        if cost < min_train_cost:
            min_train_cost = cost
            best_tau = tau

    cost_tau = best_tau
    cost_train_at_tau = min_train_cost

    # Test costs
    def compute_cost_metrics(y_t, p_t, tau_val):
        pred = (p_t >= tau_val).astype(int)
        tp, fp, tn, fn = get_metrics(y_t, pred)[:4]
        return 50 * fp + 400 * fn

    cost_test_at_tau = compute_cost_metrics(y_test, p_test, cost_tau)
    cost_test_at_0_5 = compute_cost_metrics(y_test, p_test, 0.5)
    cost_test_flag_nobody = 50 * 0 + 400 * int(np.sum(y_test == 1))

    # Write results
    results = [
        f"train_rows = {train_rows}",
        f"train_positives = {train_positives}",
        f"test_rows = {test_rows}",
        f"test_positives = {test_positives}",
        f"loss_epoch_0 = {loss_epoch_0:.4f}",
        f"loss_epoch_1 = {loss_epoch_1:.4f}",
        f"loss_epoch_10 = {loss_epoch_10:.4f}",
        f"loss_epoch_100 = {loss_epoch_100:.4f}",
        f"loss_epoch_1000 = {loss_epoch_1000:.4f}",
        f"weight_X1 = {w_fin[0]:.4f}",
        f"weight_X2 = {w_fin[1]:.4f}",
        f"weight_X3 = {w_fin[2]:.4f}",
        f"weight_X4 = {w_fin[3]:.4f}",
        f"weight_X5 = {w_fin[4]:.4f}",
        f"weight_X6 = {w_fin[5]:.4f}",
        f"weight_X7 = {w_fin[6]:.4f}",
        f"bias = {b_fin:.4f}",
        f"test_bce = {test_bce:.4f}",
        f"loss_epoch_200_eta_0.01 = {loss_epoch_200_eta_001:.4f}",
        f"loss_epoch_200_eta_0.1 = {loss_epoch_200_eta_01:.4f}",
        f"loss_epoch_200_eta_1.0 = {loss_epoch_200_eta_10:.4f}",
        f"odds_ratio_X1 = {odds_ratios[0]:.4f}",
        f"odds_ratio_X2 = {odds_ratios[1]:.4f}",
        f"odds_ratio_X3 = {odds_ratios[2]:.4f}",
        f"odds_ratio_X4 = {odds_ratios[3]:.4f}",
        f"odds_ratio_X5 = {odds_ratios[4]:.4f}",
        f"odds_ratio_X6 = {odds_ratios[5]:.4f}",
        f"odds_ratio_X7 = {odds_ratios[6]:.4f}",
        f"tau_0.5_tp = {tp_05}",
        f"tau_0.5_fp = {fp_05}",
        f"tau_0.5_tn = {tn_05}",
        f"tau_0.5_fn = {fn_05}",
        f"tau_0.5_accuracy = {acc_05:.4f}",
        f"tau_0.5_precision = {prec_05:.4f}",
        f"tau_0.5_recall = {rec_05:.4f}",
        f"tau_0.5_f1 = {f1_05:.4f}",
        f"tau_0.5_specificity = {spec_05:.4f}",
        f"tau_0.3_tp = {tp_03}",
        f"tau_0.3_fp = {fp_03}",
        f"tau_0.3_tn = {tn_03}",
        f"tau_0.3_fn = {fn_03}",
        f"tau_0.3_accuracy = {acc_03:.4f}",
        f"tau_0.3_precision = {prec_03:.4f}",
        f"tau_0.3_recall = {rec_03:.4f}",
        f"tau_0.3_f1 = {f1_03:.4f}",
        f"tau_0.3_specificity = {spec_03:.4f}",
        f"blind_accuracy = {blind_accuracy:.4f}",
        f"knn_best_k = {knn_best_k}",
        f"knn_tp = {knn_tp}",
        f"knn_fp = {knn_fp}",
        f"knn_tn = {knn_tn}",
        f"knn_fn = {knn_fn}",
        f"knn_accuracy = {knn_acc:.4f}",
        f"knn_precision = {knn_prec:.4f}",
        f"knn_recall = {knn_rec:.4f}",
        f"knn_f1 = {knn_f1:.4f}",
        f"cost_tau = {cost_tau:.4f}",
        f"cost_train_at_tau = {min_train_cost}",
        f"cost_test_at_tau = {cost_test_at_tau}",
        f"cost_test_at_0.5 = {cost_test_at_0_5}",
        f"cost_test_flag_nobody = {cost_test_flag_nobody}"
    ]

    with open("outputs/part2_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")
    print("Part 2 completed successfully.")

if __name__ == "__main__":
    run_part2()