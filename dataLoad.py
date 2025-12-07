import os
import re
import pandas as pd

# === Configuration ===
ROOT_DIR = "CSV"   # root directory containing subfolders
TRAIN_OUTPUT = "awid3_train.csv"
TEST_OUTPUT = "awid3_test.csv"


def get_folder_index(folder_name: str) -> int:
    """
    Extract the numeric prefix from a folder name.
    Example: "1.Deauth" -> 1, "2.Disas" -> 2
    If no prefix is found, place it at the end.
    """
    m = re.match(r"(\d+)", folder_name)
    if m:
        return int(m.group(1))
    return 10**9


def get_file_index(file_name: str) -> int:
    """
    Extract the numeric index from CSV file names.
    Example: "Deauth_0.csv" -> 0, "Deauth_12.csv" -> 12
    """
    m = re.search(r"(\d+)(?=\.csv$)", file_name)
    if m:
        return int(m.group(1))
    return 10**9


def split_folder_df(df: pd.DataFrame, normal_label: str = "Normal", train_ratio: float = 0.6):
    """
    Split data inside a single subfolder into train/test with conditions:
    - Total samples N → target train size n_train = int(N * train_ratio)
    - Train set MUST contain only rows with label == normal_label
    - Scan from the beginning, sequentially picking Normal rows until n_train is reached
    - If earlier rows contain abnormal samples, they are skipped
    - Test set consists of all remaining rows (order preserved)
    """
    total_n = len(df)
    n_train = int(total_n * train_ratio)
    label_col = df.columns[-1]  # last column = label

    train_indices = []

    # Collect the first n_train Normal samples (in original order)
    for idx, label in enumerate(df[label_col]):
        if len(train_indices) >= n_train:
            break
        if label == normal_label:
            train_indices.append(idx)

    if len(train_indices) < n_train:
        print(
            f"Warning: Not enough Normal samples to reach {train_ratio*100:.1f}% "
            f"train size. Collected {len(train_indices)} / target {n_train}."
        )

    # Create train/test sets
    train_mask = df.index.isin(train_indices)
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    return train_df, test_df


def main():
    # Collect all subfolders and sort by numeric prefix
    subfolders = [
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ]
    subfolders_sorted = sorted(subfolders, key=get_folder_index)

    all_train_parts = []
    all_test_parts = []

    for subfolder in subfolders_sorted:
        subfolder_path = os.path.join(ROOT_DIR, subfolder)
        print(f"Processing subfolder: {subfolder_path}")

        # List all CSV files
        csv_files = [
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(".csv")
        ]
        # Sort by numeric suffix
        csv_files_sorted = sorted(csv_files, key=get_file_index)

        dfs = []
        # Load each CSV in sorted order
        for csv_file in csv_files_sorted:
            csv_path = os.path.join(subfolder_path, csv_file)
            print(f"  Reading file: {csv_path}")
            df = pd.read_csv(csv_path)
            dfs.append(df)

        if not dfs:
            print(f"  Warning: No CSV files found in {subfolder_path}. Skipping.")
            continue

        folder_df = pd.concat(dfs, ignore_index=True)
        print(f"  Total samples in this folder: {len(folder_df)}")

        # Perform 60/40 split with train containing only Normal samples
        folder_train_df, folder_test_df = split_folder_df(folder_df, normal_label="Normal", train_ratio=0.6)
        print(
            f"  Split result → train: {len(folder_train_df)}, "
            f"test: {len(folder_test_df)}, total: {len(folder_df)}"
        )

        all_train_parts.append(folder_train_df)
        all_test_parts.append(folder_test_df)

    # Combine all folder-level results
    if all_train_parts:
        global_train_df = pd.concat(all_train_parts, ignore_index=True)
    else:
        global_train_df = pd.DataFrame()

    if all_test_parts:
        global_test_df = pd.concat(all_test_parts, ignore_index=True)
    else:
        global_test_df = pd.DataFrame()

    print(f"\nFinal train set size: {len(global_train_df)}")
    print(f"Final test set size: {len(global_test_df)}")

    # Save final CSV outputs
    global_train_df.to_csv(TRAIN_OUTPUT, index=False)
    global_test_df.to_csv(TEST_OUTPUT, index=False)

    print(f"\nSaved train set to: {TRAIN_OUTPUT}")
    print(f"Saved test set to: {TEST_OUTPUT}")


if __name__ == "__main__":
    main()
