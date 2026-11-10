#Part 3
import os
import csv
import numpy as np

def run_part3():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    data = []
    with open("data/records.csv", "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            data.append([float(x) for x in parts[1:]])
    data = np.array(data)

    X_cont = data[:, :5] # X1 to X5
    outcomes = data[:, 7]
    values = data[:, 8]

    # Standardize X1 to X5 over all 800 rows
    mean_all = np.mean(X_cont, axis=0)
    sd_all = np.std(X_cont, axis=0, ddof=1)
    X_std = (X_cont - mean_all) / sd_all

    n_rows = len(X_std)

    # 3.1 K-means from scratch
    def run_kmeans(K):
        # Starting centroids: rows with indices 0, floor(800/K), ...
        centroid_indices = [i * (n_rows // K) for i in range(K)]
        centroids = X_std[centroid_indices].copy()

        assignments = np.zeros(n_rows, dtype=int)
        passes = 0

        while True:
            passes += 1
            # Assign to nearest centroid by squared Euclidean distance (tie to lower index)
            new_assignments = np.zeros(n_rows, dtype=int)
            for i in range(n_rows):
                dists = np.sum((centroids - X_std[i]) ** 2, axis=1)
                new_assignments[i] = np.argmin(dists)

            if np.all(assignments == new_assignments):
                break
            assignments = new_assignments

            # Update centroids
            for k in range(K):
                members = np.where(assignments == k)[0]
                if len(members) > 0:
                    centroids[k] = np.mean(X_std[members], axis=0)

        # WCSS
        wcss = 0.0
        for i in range(n_rows):
            c_idx = assignments[i]
            wcss += np.sum((X_std[i] - centroids[c_idx]) ** 2)

        return wcss, assignments, centroids

    wcss_results = {}
    assignments_dict = {}
    for k in [2, 3, 4, 5, 6]:
        wcss, assgn, _ = run_kmeans(k)
        wcss_results[k] = wcss
        assignments_dict[k] = assgn

    wcss_k2 = wcss_results[2]
    wcss_k6 = wcss_results[6]

    # 3.2 Choose K with elbow
    # smallest K in {3, 4, 5} for which WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = 3
    for k in [3, 4, 5]:
        diff_curr = wcss_results[k] - wcss_results[k+1]
        diff_prev = wcss_results[k-1] - wcss_results[k]
        if diff_curr < 0.5 * diff_prev:
            elbow_k = k
            break

    # Use K=3 for rest of Part 3
    k3_assgn = assignments_dict[3]

    # 3.3 Profile K=3 segments
    cluster_means_x3 = []
    for k in range(3):
        idxs = np.where(k3_assgn == k)[0]
        cluster_means_x3.append((np.mean(X_cont[idxs, 2]), k)) # X3 is index 2

    cluster_means_x3.sort(key=lambda x: x[0])
    segment_map = {cluster_means_x3[0][1]: 'A', cluster_means_x3[1][1]: 'B', cluster_means_x3[2][1]: 'C'}

    segments_data = []
    sizes = {}
    seg_records = {'A': [], 'B': [], 'C': []}

    for orig_k, seg_name in sorted(segment_map.items(), key=lambda x: x[1]):
        idxs = np.where(k3_assgn == orig_k)[0]
        seg_records[seg_name] = idxs
        size = len(idxs)
        sizes[seg_name] = size
        mean_x1 = np.mean(X_cont[idxs, 0])
        mean_x2 = np.mean(X_cont[idxs, 1])
        mean_x3 = np.mean(X_cont[idxs, 2])
        mean_x4 = np.mean(X_cont[idxs, 3])
        mean_x5 = np.mean(X_cont[idxs, 4])
        outcome_rate = np.mean(outcomes[idxs])
        mean_val = np.mean(values[idxs])
        segments_data.append([seg_name, size, mean_x1, mean_x2, mean_x3, mean_x4, mean_x5, outcome_rate, mean_val])

    with open("outputs/part3_segments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"])
        writer.writerows(segments_data)

    size_segment_A = sizes['A']
    size_segment_B = sizes['B']
    size_segment_C = sizes['C']

    # 3.4 Hierarchical clustering on supplied dissimilarity table
    dissim_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])
    cohorts = ["E1", "E2", "E3", "E4", "E5", "E6"]

    def run_hierarchical(linkage_type):
        clusters = {c: [c] for c in cohorts}
        dist_mat = dissim_matrix.copy()
        cluster_keys = list(clusters.keys())
        merges = []

        for step in range(1, 6):
            min_dist = float('inf')
            pair = (None, None)

            n_c = len(cluster_keys)
            for i in range(n_c):
                for j in range(i + 1, n_c):
                    c1 = cluster_keys[i]
                    c2 = cluster_keys[j]
                    mem1 = clusters[c1]
                    mem2 = clusters[c2]

                    # Compute linkage distance
                    distances = []
                    for m1 in mem1:
                        idx1 = cohorts.index(m1)
                        for m2 in mem2:
                            idx2 = cohorts.index(m2)
                            distances.append(dissim_matrix[idx1, idx2])

                    if linkage_type == 'single':
                        d = min(distances)
                    elif linkage_type == 'complete':
                        d = max(distances)
                    else: # average
                        d = np.mean(distances)

                    if d < min_dist:
                        min_dist = d
                        pair = (c1, c2)

            c1, c2 = pair
            # cluster_a is the one whose smallest member number is smaller
            mem1_sorted = sorted(clusters[c1], key=lambda x: int(x[1]))
            mem2_sorted = sorted(clusters[c2], key=lambda x: int(x[1]))

            new_cluster_name = "+".join(sorted(clusters[c1] + clusters[c2], key=lambda x: int(x[1])))

            if int(mem1_sorted[0][1]) < int(mem2_sorted[0][1]):
                clust_a, clust_b = c1, c2
            else:
                clust_a, clust_b = c2, c1

            merges.append([linkage_type, step, clusters[clust_a], clusters[clust_b], min_dist])

            # Update clusters and matrix
            clusters[new_cluster_name] = clusters[c1] + clusters[c2]
            del clusters[c1]
            del clusters[c2]
            cluster_keys = list(clusters.keys())

        return merges

    hier_rows = []
    all_merges = {}
    for l_type in ['single', 'complete', 'average']:
        merges = run_hierarchical(l_type)
        all_merges[l_type] = merges
        for m in merges:
            a_str = "+".join(sorted(m[2], key=lambda x: int(x[1])))
            b_str = "+".join(sorted(m[3], key=lambda x: int(x[1])))
            hier_rows.append([m[0], m[1], a_str, b_str, m[4]])

    with open("outputs/part3_hierarchical.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        writer.writerows(hier_rows)

    def get_clusters_at_cut(merges):
        heights = [m[4] for m in merges]
        gaps = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        max_gap_idx = np.argmax(gaps)
        # number of clusters = 6 - number of merges below cut (merges before max_gap_idx + 1)
        num_clusters = 6 - (max_gap_idx + 1)
        return num_clusters

    single_clusters_at_cut = get_clusters_at_cut(all_merges['single'])
    complete_clusters_at_cut = get_clusters_at_cut(all_merges['complete'])
    average_clusters_at_cut = get_clusters_at_cut(all_merges['average'])

    # 3.5 PCA by seven steps
    # (1) mean vector is zero (standardized)
    # (2) centered data is X_std
    # (3) covariance matrix S = (1 / (n-1)) * X_std^T @ X_std
    cov_matrix = (1.0 / (n_rows - 1)) * (X_std.T @ X_std)
    # (4) eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    # (5) sort by eigenvalue largest first
    sort_eig_idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sort_eig_idx]
    eigenvecs = eigenvecs[:, sort_eig_idx]

    # Sign convention: flip eigenvector so entry of largest absolute value is positive
    for col in range(eigenvecs.shape[1]):
        max_abs_idx = np.argmax(np.abs(eigenvecs[:, col]))
        if eigenvecs[max_abs_idx, col] < 0:
            eigenvecs[:, col] *= -1.0

    eigenvalue_sum = np.sum(eigenvals)
    explained_ratios = eigenvals / eigenvalue_sum
    cum_ratios = np.cumsum(explained_ratios)

    components_for_80pct = int(np.where(cum_ratios >= 0.8)[0][0] + 1)

    # Keep top 2 eigenvectors
    W = eigenvecs[:, :2]
    Z = X_std @ W

    pc1_loadings = eigenvecs[:, 0]
    pc2_loadings = eigenvecs[:, 1]

    row1_pc1, row1_pc2 = Z[0, 0], Z[0, 1]
    row2_pc1, row2_pc2 = Z[1, 0], Z[1, 1]
    row3_pc1, row3_pc2 = Z[2, 0], Z[2, 1]

    mean_pcl_segment_A = np.mean(Z[seg_records['A'], 0])
    mean_pcl_segment_B = np.mean(Z[seg_records['B'], 0])
    mean_pcl_segment_C = np.mean(Z[seg_records['C'], 0])

    # Write results
    results = [
        f"wcss_k2 = {wcss_k2:.4f}",
        f"wcss_k3 = {wcss_results[3]:.4f}",
        f"wcss_k4 = {wcss_results[4]:.4f}",
        f"wcss_k5 = {wcss_results[5]:.4f}",
        f"wcss_k6 = {wcss_k6:.4f}",
        f"elbow_k = {elbow_k}",
        f"size_segment_A = {size_segment_A}",
        f"size_segment_B = {size_segment_B}",
        f"size_segment_C = {size_segment_C}",
        f"single_clusters_at_cut = {single_clusters_at_cut}",
        f"complete_clusters_at_cut = {complete_clusters_at_cut}",
        f"average_clusters_at_cut = {average_clusters_at_cut}",
        f"eigenvalue_sum = {eigenvalue_sum:.4f}",
        f"eigenvalue_1 = {eigenvals[0]:.4f}",
        f"eigenvalue_2 = {eigenvals[1]:.4f}",
        f"eigenvalue_3 = {eigenvals[2]:.4f}",
        f"eigenvalue_4 = {eigenvals[3]:.4f}",
        f"eigenvalue_5 = {eigenvals[4]:.4f}",
        f"explained_ratio_1 = {explained_ratios[0]:.4f}",
        f"explained_ratio_2 = {explained_ratios[1]:.4f}",
        f"explained_ratio_3 = {explained_ratios[2]:.4f}",
        f"explained_ratio_4 = {explained_ratios[3]:.4f}",
        f"explained_ratio_5 = {explained_ratios[4]:.4f}",
        f"cumulative_ratio_1 = {cum_ratios[0]:.4f}",
        f"cumulative_ratio_2 = {cum_ratios[1]:.4f}",
        f"cumulative_ratio_3 = {cum_ratios[2]:.4f}",
        f"cumulative_ratio_4 = {cum_ratios[3]:.4f}",
        f"cumulative_ratio_5 = {cum_ratios[4]:.4f}",
        f"components_for_80pct = {components_for_80pct}",
        f"pcl_loading_X1 = {pc1_loadings[0]:.4f}",
        f"pcl_loading_X2 = {pc1_loadings[1]:.4f}",
        f"pcl_loading_X3 = {pc1_loadings[2]:.4f}",
        f"pcl_loading_X4 = {pc1_loadings[3]:.4f}",
        f"pcl_loading_X5 = {pc1_loadings[4]:.4f}",
        f"pc2_loading_X1 = {pc2_loadings[0]:.4f}",
        f"pc2_loading_X2 = {pc2_loadings[1]:.4f}",
        f"pc2_loading_X3 = {pc2_loadings[2]:.4f}",
        f"pc2_loading_X4 = {pc2_loadings[3]:.4f}",
        f"pc2_loading_X5 = {pc2_loadings[4]:.4f}",
        f"row1_pc1 = {row1_pc1:.4f}",
        f"row1_pc2 = {row1_pc2:.4f}",
        f"row2_pc1 = {row2_pc1:.4f}",
        f"row2_pc2 = {row2_pc2:.4f}",
        f"row3_pc1 = {row3_pc1:.4f}",
        f"row3_pc2 = {row3_pc2:.4f}",
        f"mean_pcl_segment_A = {mean_pcl_segment_A:.4f}",
        f"mean_pcl_segment_B = {mean_pcl_segment_B:.4f}",
        f"mean_pcl_segment_C = {mean_pcl_segment_C:.4f}"
    ]

    with open("outputs/part3_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")
    print("Part 3 completed successfully.")

if __name__ == "__main__":
    run_part3()