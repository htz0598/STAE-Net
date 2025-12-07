import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support


def load_and_merge_scores(
    resae_train_path="resae_scores_train.csv",
    resae_test_path="resae_scores_test.csv",
    tmae_train_path="tmae_scores_train.csv",
    tmae_test_path="tmae_scores_test.csv",
):
    """
    Load ResAE and TMAE scores and merge them by idx
    for both train and test splits.
    """
    # Load CSVs
    res_train = pd.read_csv(resae_train_path)
    res_test = pd.read_csv(resae_test_path)

    tmae_train = pd.read_csv(tmae_train_path)
    tmae_test = pd.read_csv(tmae_test_path)

    # Merge by idx (inner join) for train
    train_merged = pd.merge(
        res_train,
        tmae_train,
        on="idx",
        suffixes=("_img", "_seq"),
        how="inner",
    )

    # Align label columns
    if "label_img" in train_merged.columns and "label_seq" in train_merged.columns:
        mismatch = (train_merged["label_img"] != train_merged["label_seq"]).sum()
        if mismatch > 0:
            print(f"Warning: {mismatch} label mismatches in train merge.")
        train_merged["label"] = train_merged["label_img"]
        train_merged = train_merged.drop(columns=["label_img", "label_seq"])
    else:
        if "label" not in train_merged.columns:
            raise ValueError("Cannot find label columns in merged train data.")

    # Same for test
    test_merged = pd.merge(
        res_test,
        tmae_test,
        on="idx",
        suffixes=("_img", "_seq"),
        how="inner",
    )

    if "label_img" in test_merged.columns and "label_seq" in test_merged.columns:
        mismatch = (test_merged["label_img"] != test_merged["label_seq"]).sum()
        if mismatch > 0:
            print(f"Warning: {mismatch} label mismatches in test merge.")
        test_merged["label"] = test_merged["label_img"]
        test_merged = test_merged.drop(columns=["label_img", "label_seq"])
    else:
        if "label" not in test_merged.columns:
            raise ValueError("Cannot find label columns in merged test data.")

    # Rename score columns to unified names (if needed)
    if "E_img" not in train_merged.columns:
        raise ValueError("Column 'E_img' not found in train data.")
    if "E_seq" not in train_merged.columns:
        raise ValueError("Column 'E_seq' not found in train data.")

    if "E_img" not in test_merged.columns or "E_seq" not in test_merged.columns:
        raise ValueError("Columns 'E_img'/'E_seq' not found in test data.")

    # Sort by idx just for cleanliness
    train_merged = train_merged.sort_values(by="idx").reset_index(drop=True)
    test_merged = test_merged.sort_values(by="idx").reset_index(drop=True)

    return train_merged, test_merged


def compute_lambda(train_df: pd.DataFrame, normal_label="Normal"):
    """
    Compute adaptive fusion weight lambda using training data.

    Following the paper:
      - Only Normal samples in the training set should be used.
      - sigma_img = std(E_img)
      - sigma_seq = std(E_seq)
      - lambda = sigma_seq / (sigma_img + sigma_seq)
    """
    normal_df = train_df[train_df["label"] == normal_label].copy()
    if len(normal_df) == 0:
        print("Warning: No Normal samples found in training set; using all samples to compute lambda.")
        normal_df = train_df

    sigma_img = normal_df["E_img"].std(ddof=0)
    sigma_seq = normal_df["E_seq"].std(ddof=0)

    denom = sigma_img + sigma_seq
    if denom == 0:
        print("Warning: sigma_img + sigma_seq == 0, fallback lambda = 0.5")
        lam = 0.5
    else:
        lam = float(sigma_seq / denom)

    print(f"Computed lambda = {lam:.6f} (sigma_img={sigma_img:.6e}, sigma_seq={sigma_seq:.6e})")
    return lam


def compute_fusion_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, lam: float):
    """
    Compute fusion anomaly scores:
      A_t = lambda * E_img + (1 - lambda) * E_seq
    for both train and test sets.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["A_t"] = lam * train_df["E_img"] + (1.0 - lam) * train_df["E_seq"]
    test_df["A_t"] = lam * test_df["E_img"] + (1.0 - lam) * test_df["E_seq"]

    return train_df, test_df


def select_threshold(train_df: pd.DataFrame, normal_label="Normal", percentile=99.0):
    """
    Select anomaly threshold tau based on training Normal samples.
    Use percentile of A_t (e.g., 99th percentile).
    """
    normal_df = train_df[train_df["label"] == normal_label].copy()
    if len(normal_df) == 0:
        print("Warning: No Normal samples found for threshold selection; using all samples.")
        normal_df = train_df

    tau = float(np.percentile(normal_df["A_t"].values, percentile))
    print(f"Selected threshold tau = {tau:.6f} at {percentile}th percentile.")
    return tau


def apply_threshold(df: pd.DataFrame, tau: float):
    """
    Apply threshold tau to assign binary predictions:
      pred = 0 (Normal) if A_t <= tau
      pred = 1 (Anomaly) if A_t > tau
    """
    df = df.copy()
    df["pred"] = (df["A_t"] > tau).astype(int)
    return df


def save_fusion_scores(path: str, df: pd.DataFrame):
    """
    Save fusion scores to CSV with columns:
      idx, label, E_img, E_seq, A_t, pred
    """
    cols = ["idx", "label", "E_img", "E_seq", "A_t", "pred"]
    df_to_save = df[cols].copy()
    df_to_save.to_csv(path, index=False)
    print(f"Saved fusion scores to: {path}")


def compute_metrics_on_test(test_df: pd.DataFrame, normal_label="Normal"):
    """
    Compute ROC AUC, Precision, Recall, F1 on the TEST set.
    Positive class = anomaly (label != normal_label).
    """
    # Ground truth: anomaly = 1, normal = 0
    y_true = (test_df["label"] != normal_label).astype(int).values
    y_score = test_df["A_t"].values
    y_pred = test_df["pred"].astype(int).values

    # ROC AUC
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        print("Warning: Only one class present in y_true for test set; ROC AUC is undefined.")
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_score)

    # Precision, Recall, F1 (binary, positive = anomaly)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )

    return roc_auc, precision, recall, f1


def main():
    # Paths of branch scores
    resae_train_path = "resae_scores_train.csv"
    resae_test_path = "resae_scores_test.csv"
    tmae_train_path = "tmae_scores_train.csv"
    tmae_test_path = "tmae_scores_test.csv"

    # Output paths
    fusion_train_path = "fusion_scores_train.csv"
    fusion_test_path = "fusion_scores_test.csv"
    metrics_path = "fusion_metrics.txt"

    print("Loading and merging ResAE/TMAE scores...")
    train_df, test_df = load_and_merge_scores(
        resae_train_path=resae_train_path,
        resae_test_path=resae_test_path,
        tmae_train_path=tmae_train_path,
        tmae_test_path=tmae_test_path,
    )

    print(f"Merged train samples: {len(train_df)}")
    print(f"Merged test samples: {len(test_df)}")

    # Compute adaptive fusion weight lambda on training Normal samples
    lam = compute_lambda(train_df, normal_label="Normal")

    # Compute fusion scores A_t
    print("Computing fusion scores A_t for train and test...")
    train_df, test_df = compute_fusion_scores(train_df, test_df, lam)

    # Select threshold tau on training Normal samples (99th percentile)
    tau = select_threshold(train_df, normal_label="Normal", percentile=99.0)

    # Apply threshold to get predictions
    print("Applying threshold to obtain final predictions...")
    train_df = apply_threshold(train_df, tau)
    test_df = apply_threshold(test_df, tau)

    # Save fusion scores (optional but usually useful)
    save_fusion_scores(fusion_train_path, train_df)
    save_fusion_scores(fusion_test_path, test_df)

    # Compute metrics on TEST set
    print("Computing metrics on test set...")
    roc_auc, precision, recall, f1 = compute_metrics_on_test(test_df, normal_label="Normal")

    print("\n===== Evaluation on TEST set =====")
    print(f"ROC AUC   : {roc_auc:.6f}" if not np.isnan(roc_auc) else "ROC AUC   : NaN (only one class present)")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-score  : {f1:.6f}")

    # Save metrics to a text file
    with open(metrics_path, "w") as f:
        if np.isnan(roc_auc):
            f.write("ROC AUC   : NaN (only one class present in y_true)\n")
        else:
            f.write(f"ROC AUC   : {roc_auc:.6f}\n")
        f.write(f"Precision : {precision:.6f}\n")
        f.write(f"Recall    : {recall:.6f}\n")
        f.write(f"F1-score  : {f1:.6f}\n")
        f.write(f"lambda    : {lam:.8f}\n")
        f.write(f"tau       : {tau:.8f}\n")

    print(f"\nSaved evaluation metrics to: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
