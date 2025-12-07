import pandas as pd
import numpy as np


class AWID3Preprocessor:
    def __init__(self, target_dim=256):
        self.target_dim = target_dim

        # Will be filled after fit()
        self.label_col = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.cat_maps_ = {}        # col -> {category: code}
        self.feature_cols_ = None  # all feature columns after padding
        self.min_ = None           # Series: min per feature (train)
        self.max_ = None           # Series: max per feature (train)
        self.denom_ = None         # Series: (max - min), with zeros handled
        self.permutation_ = None   # ordered feature names by variance (desc)

    def _split_features_labels(self, df: pd.DataFrame):
        """
        Split a DataFrame into features (X) and labels (y).
        Assumes the last column is the label.
        """
        self.label_col = df.columns[-1]
        X = df.iloc[:, :-1].copy()
        y = df[self.label_col].copy()
        return X, y

    def _detect_types(self, X: pd.DataFrame):
        """
        Detect numeric and categorical columns based on pandas dtypes.
        Numeric: int/float
        Categorical: object/string and others that are not numeric.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        self.numeric_cols_ = numeric_cols
        self.categorical_cols_ = categorical_cols

    def _encode_categorical_train(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encode categorical columns on the TRAIN set.
        Store mappings to be reused on test.
        """
        X_enc = X.copy()
        for col in self.categorical_cols_:
            # Convert to string to avoid problems with mixed types
            values = X_enc[col].astype("string")
            # factorize returns codes and unique categories
            codes, uniques = pd.factorize(values, sort=True)
            mapping = {cat: int(code) for code, cat in enumerate(uniques)}
            self.cat_maps_[col] = mapping
            X_enc[col] = codes.astype(float)  # use float for normalization later
        return X_enc

    def _encode_categorical_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored label encodings to TEST (or any new) data.
        Unseen categories are mapped to -1.
        """
        X_enc = X.copy()
        for col in self.categorical_cols_:
            values = X_enc[col].astype("string")
            mapping = self.cat_maps_.get(col, {})
            # Map known categories; unseen categories become NaN, then filled with -1
            X_enc[col] = values.map(mapping).fillna(-1).astype(float)
        return X_enc

    def _pad_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Zero-pad feature dimensions to self.target_dim.
        Padding columns are added at the end with all zeros.
        """
        X_pad = X.copy()
        n_features = X_pad.shape[1]
        if n_features > self.target_dim:
            raise ValueError(
                f"Current feature dimension ({n_features}) is larger than target_dim ({self.target_dim})."
            )

        n_pad = self.target_dim - n_features
        for i in range(n_pad):
            pad_name = f"_pad_{i}"
            X_pad[pad_name] = 0.0

        return X_pad

    def fit(self, train_df: pd.DataFrame):
        """
        Fit preprocessing parameters on the TRAIN set only:
        - type detection
        - categorical encoding mapping
        - zero padding to target_dim
        - min/max for normalization
        - variance-based feature ordering
        """
        # Split features and labels
        X_train, y_train = self._split_features_labels(train_df)

        # Detect numeric vs categorical
        self._detect_types(X_train)

        # Encode categoricals on train
        X_train_enc = self._encode_categorical_train(X_train)

        # Ensure numeric columns are float
        for col in self.numeric_cols_:
            X_train_enc[col] = pd.to_numeric(X_train_enc[col], errors="coerce")

        # Zero padding
        X_train_pad = self._pad_features(X_train_enc)
        self.feature_cols_ = X_train_pad.columns.tolist()

        # Fill NaNs with 0 before computing min/max
        X_train_pad = X_train_pad.fillna(0.0)

        # Compute min/max on TRAIN ONLY
        self.min_ = X_train_pad.min(axis=0)
        self.max_ = X_train_pad.max(axis=0)
        denom = self.max_ - self.min_
        # Avoid division by zero
        denom_replaced = denom.replace(0, 1.0)
        self.denom_ = denom_replaced

        # Normalize train
        X_train_norm = (X_train_pad - self.min_) / self.denom_

        # Compute variance on normalized TRAIN
        variances = X_train_norm.var(axis=0)

        # Sort feature names by variance (descending)
        self.permutation_ = variances.sort_values(ascending=False).index.tolist()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessing pipeline to ANY data (train or test):
        - encode categoricals with stored mapping
        - zero padding to same feature set
        - normalization using TRAIN stats (min_/denom_)
        - feature reordering by TRAIN variance order
        """
        if self.feature_cols_ is None or self.permutation_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        # Split features and labels
        label_col = df.columns[-1]
        X = df.iloc[:, :-1].copy()
        y = df[label_col].copy()

        # Encode categoricals with stored mapping
        X_enc = X.copy()
        if self.categorical_cols_ is not None:
            X_enc = self._encode_categorical_test(X_enc)

        # Ensure numeric columns are float
        if self.numeric_cols_ is not None:
            for col in self.numeric_cols_:
                if col in X_enc.columns:
                    X_enc[col] = pd.to_numeric(X_enc[col], errors="coerce")

        # Reindex to match feature_cols_ (original+pad), missing columns filled with 0
        X_enc = X_enc.reindex(columns=self.feature_cols_, fill_value=0.0)

        # Fill NaNs with 0 before normalization
        X_enc = X_enc.fillna(0.0)

        # Normalize using TRAIN stats
        X_norm = (X_enc - self.min_) / self.denom_

        # Reorder features by variance-based permutation
        X_reordered = X_norm[self.permutation_].copy()

        # Optional: rename columns to generic names (f0..f{D-1}) for clarity
        new_feature_names = [f"f{i}" for i in range(len(X_reordered.columns))]
        X_reordered.columns = new_feature_names

        # Append label as the last column
        X_reordered[label_col] = y.values

        return X_reordered


def main():
    # Input and output file paths
    train_input_path = "awid3_train.csv"
    test_input_path = "awid3_test.csv"

    train_output_path = "awid3_train_preprocessed.csv"
    test_output_path = "awid3_test_preprocessed.csv"

    print("Loading train and test CSV files...")
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    print("Fitting preprocessor on train set...")
    preproc = AWID3Preprocessor(target_dim=256)
    preproc.fit(train_df)

    print("Transforming train set...")
    train_processed = preproc.transform(train_df)
    print("Transforming test set...")
    test_processed = preproc.transform(test_df)

    print(f"Saving preprocessed train set to: {train_output_path}")
    train_processed.to_csv(train_output_path, index=False)

    print(f"Saving preprocessed test set to: {test_output_path}")
    test_processed.to_csv(test_output_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
