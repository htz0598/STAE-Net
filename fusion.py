import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def _coalesce_label(row):
    """Prefer label_img, else label_seq."""
    if pd.notna(row.get("label_img", np.nan)):
        return row["label_img"]
    if pd.notna(row.get("label_seq", np.nan)):
        return row["label_seq"]
    return np.nan


def load_and_align_scores_paper(
    resae_train_path="resae_scores_train.csv",
    resae_test_path="resae_scores_test.csv",
    tmae_train_path="tmae_scores_train.csv",
    tmae_test_path="tmae_scores_test.csv",
):
    """
    Paper-aligned alignment:
    - Use ResAE scores as the time index backbone (per-frame idx).
    - Left-join TMAE scores (per-segment end_idx) onto ResAE by idx.
      => early k<T points keep E_img but have missing E_seq (NaN).
    """
    res_train = pd.read_csv(resae_train_path)
    res_test = pd.read_csv(resae_test_path)
    tmae_train = pd.read_csv(tmae_train_path)
    tmae_test = pd.read_csv(tmae_test_path)

    # Validate required columns
    for df, name in [(res_train, "res_train"), (res_test, "res_test")]:
        for col in ["idx", "label", "E_img"]:
            if col not in df.columns:
                raise ValueError(f"{name} missing required column: {col}")
    for df, name in [(tmae_train, "tmae_train"), (tmae_test, "tmae_test")]:
        for col in ["idx", "label", "E_seq"]:
            if col not in df.columns:
                raise ValueError(f"{name} missing required column: {col}")

    # Rename label columns so we can compare
    res_train = res_train.rename(columns={"label": "label_img"})
    res_test = res_test.rename(columns={"label": "label_img"})
    tmae_train = tmae_train.rename(columns={"label": "label_seq"})
    tmae_test = tmae_test.rename(columns={"label": "label_seq"})

    # Left join: keep all ResAE frames
    train = pd.merge(res_train, tmae_train[["idx", "label_seq", "E_seq"]], on="idx", how="left")
    test = pd.merge(res_test, tmae_test[["idx", "label_seq", "E_seq"]], on="idx", how="left")

    # Build unified label
    train["label"] = train.apply(_coalesce_label, axis=1)
    test["label"] = test.apply(_coalesce_label, axis=1)

    # Optional: warn label mismatch where both exist
    train_mismatch = ((train["label_img"].notna()) & (train["label_seq"].notna()) & (train["label_img"] != train["label_seq"])).sum()
    test_mismatch = ((test["label_img"].notna()) & (test["label_seq"].notna()) & (test["label_img"] != test["label_seq"])).sum()
    if train_mismatch > 0:
        print(f"Warning: {train_mismatch} label mismatches in TRAIN (idx where both branches exist).")
    if test_mismatch > 0:
        print(f"Warning: {test_mismatch} label mismatches in TEST (idx where both branches exist).")

    # Keep only needed columns, sort by idx
    train = train[["idx", "label", "E_img", "E_seq"]].sort_values("idx").reset_index(drop=True)
    test = test[["idx", "label", "E_img", "E_seq"]].sort_values("idx").reset_index(drop=True)

    # Drop rows with missing label (should not happen; but safer)
    train = train.dropna(subset=["label"]).reset_index(drop=True)
    test = test.dropna(subset=["label"]).reset_index(drop=True)

    return train, test


def compute_lambda_paper(train_df: pd.DataFrame, normal_label="Normal"):
    """
    Paper-aligned lambda:
      sigma_img = std(E_img) on Normal train frames
      sigma_seq = std(E_seq) on Normal train frames where E_seq exists
      lambda = sigma_seq / (sigma_img + sigma_seq)
    """
    normal = train_df[train_df["label"] == normal_label].copy()
    if len(normal) == 0:
        print("Warning: No Normal samples in train; using all samples for lambda.")
        normal = train_df

    sigma_img = float(normal["E_img"].std(ddof=0))

    normal_seq = normal.dropna(subset=["E_seq"])
    if len(normal_seq) == 0:
        print("Warning: No E_seq available in Normal train for lambda; fallback lambda=1.0 (image-only).")
        return 1.0

    sigma_seq = float(normal_seq["E_seq"].std(ddof=0))

    denom = sigma_img + sigma_seq
    if denom == 0:
        print("Warning: sigma_img + sigma_seq == 0; fallback lambda=0.5")
        return 0.5

    lam = float(sigma_seq / denom)
    print(f"Computed lambda = {lam:.6f} (sigma_img={sigma_img:.6e}, sigma_seq={sigma_seq:.6e})")
    return lam


def compute_fusion_scores_paper(df: pd.DataFrame, lam: float):
    """
    Paper-aligned fusion with missing E_seq handling:
      If E_seq is missing (k<T), skip temporal branch => A_t = E_img
      Else A_t = lam*E_img + (1-lam)*E_seq
    Also handles the rare case E_img missing by falling back to E_seq.
    """
    out = df.copy()

    has_img = out["E_img"].notna()
    has_seq = out["E_seq"].notna()

    out["A_t"] = np.nan

    # Both available -> fuse
    both = has_img & has_seq
    out.loc[both, "A_t"] = lam * out.loc[both, "E_img"] + (1.0 - lam) * out.loc[both, "E_seq"]

    # Only image -> image-only (k<T case)
    img_only = has_img & (~has_seq)
    out.loc[img_only, "A_t"] = out.loc[img_only, "E_img"]

    # Only seq -> seq-only fallback
    seq_only = (~has_img) & has_seq
    out.loc[seq_only, "A_t"] = out.loc[seq_only, "E_seq"]

    # If still NaN, something is wrong
    if out["A_t"].isna().any():
        n_bad = int(out["A_t"].isna().sum())
        print(f"Warning: {n_bad} rows have NaN A_t (missing both E_img and E_seq). They will be dropped.")
        out = out.dropna(subset=["A_t"]).reset_index(drop=True)

    return out


def select_threshold_paper(train_df: pd.DataFrame, normal_label="Normal", percentile=99.0):
    """
    Paper-aligned tau: percentile of A_t on Normal train samples.
    (Includes early points where A_t = E_img.)
    """
    normal = train_df[train_df["label"] == normal_label].copy()
    if len(normal) == 0:
        print("Warning: No Normal samples for tau; using all samples.")
        normal = train_df

    tau = float(np.percentile(normal["A_t"].values, percentile))
    print(f"Selected threshold tau = {tau:.6f} at {percentile}th percentile.")
    return tau


def apply_threshold(df: pd.DataFrame, tau: float):
    out = df.copy()
    out["pred"] = (out["A_t"] > tau).astype(int)
    return out


def compute_metrics(test_df: pd.DataFrame, normal_label="Normal"):
    y_true = (test_df["label"] != normal_label).astype(int).values
    y_score = test_df["A_t"].values
    y_pred = test_df["pred"].astype(int).values

    # ROC AUC (optional)
    if len(np.unique(y_true)) == 1:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return roc_auc, float(precision), float(recall), float(f1)


def save_fusion_scores(path: str, df: pd.DataFrame):
    cols = ["idx", "label", "E_img", "E_seq", "A_t", "pred"]
    df[cols].to_csv(path, index=False)
    print(f"Saved fusion scores to: {path}")


def main():
    # Inputs
    resae_train_path = "resae_scores_train.csv"
    resae_test_path = "resae_scores_test.csv"
    tmae_train_path = "tmae_scores_train.csv"
    tmae_test_path = "tmae_scores_test.csv"

    # Outputs
    fusion_train_path = "fusion_scores_train.csv"
    fusion_test_path = "fusion_scores_test.csv"
    metrics_path = "fusion_metrics.txt"

    print("Loading & aligning scores (paper-aligned; keep k<T frames)...")
    train_df, test_df = load_and_align_scores_paper(
        resae_train_path=resae_train_path,
        resae_test_path=resae_test_path,
        tmae_train_path=tmae_train_path,
        tmae_test_path=tmae_test_path,
    )
    print(f"Aligned train frames: {len(train_df)}")
    print(f"Aligned test frames : {len(test_df)}")

    lam = compute_lambda_paper(train_df, normal_label="Normal")

    print("Computing A_t fusion scores...")
    train_df = compute_fusion_scores_paper(train_df, lam)
    test_df = compute_fusion_scores_paper(test_df, lam)

    tau = select_threshold_paper(train_df, normal_label="Normal", percentile=99.0)

    print("Applying threshold...")
    train_df = apply_threshold(train_df, tau)
    test_df = apply_threshold(test_df, tau)

    save_fusion_scores(fusion_train_path, train_df)
    save_fusion_scores(fusion_test_path, test_df)

    print("Computing metrics on TEST...")
    roc_auc, precision, recall, f1 = compute_metrics(test_df, normal_label="Normal")

    print("\n===== Evaluation on TEST set =====")
    print(f"ROC AUC   : {roc_auc:.6f}" if not np.isnan(roc_auc) else "ROC AUC   : NaN (only one class present)")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-score  : {f1:.6f}")
    print(f"lambda    : {lam:.8f}")
    print(f"tau       : {tau:.8f}")

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
