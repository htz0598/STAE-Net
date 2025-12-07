import os
import numpy as np
import pandas as pd
from PIL import Image


class SRFImageBuilder:
    def __init__(self, Q=16):
        """
        Q: number of quantization states for iMTF.
           For AWID3, the paper uses Q = 16.
        """
        self.Q = Q
        self.W = None          # Global transition matrix for iMTF (Q x Q)
        self.D = None          # Feature dimension (D')
        self.feature_cols = None
        self.label_col = None

    def _extract_features_and_labels(self, df: pd.DataFrame):
        """
        Split the DataFrame into feature matrix X and labels y.
        Assumes the last column is the label.
        """
        self.label_col = df.columns[-1]
        self.feature_cols = df.columns[:-1].tolist()
        X = df[self.feature_cols].to_numpy(dtype=float)
        y = df[self.label_col].to_numpy()
        self.D = X.shape[1]
        return X, y

    def _quantize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize features into Q discrete states in [0, Q-1].
        X is assumed to be in [0, 1].
        """
        X_clipped = np.clip(X, 0.0, 1.0)
        # Uniform bins: [0, 1) → [0, Q-1]
        states = (X_clipped * self.Q).astype(int)
        states = np.clip(states, 0, self.Q - 1)
        return states

    def fit(self, train_df: pd.DataFrame):
        """
        Fit global parameters on TRAIN set only:
        - Read features and labels
        - Quantize features into Q states
        - Build global transition matrix W for iMTF
        """
        X_train, y_train = self._extract_features_and_labels(train_df)

        # Quantize train features
        states = self._quantize_features(X_train)  # shape: (N, D)

        # Build global transition matrix W (Q x Q)
        W_counts = np.zeros((self.Q, self.Q), dtype=np.float64)

        # We treat feature indices as pseudo-time indices.
        # For each sample, transitions are between consecutive features along the index axis.
        N, D = states.shape
        for n in range(N):
            s = states[n]
            # transitions: s[k] -> s[k+1]
            for k in range(D - 1):
                i = s[k]
                j = s[k + 1]
                W_counts[i, j] += 1.0

        # Normalize each row to obtain transition probabilities
        row_sums = W_counts.sum(axis=1, keepdims=True)
        # To avoid division by zero, if a row sum is zero, keep it as zeros
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.divide(W_counts, row_sums, where=(row_sums != 0))
        W = np.nan_to_num(W, nan=0.0)  # replace NaNs (rows with sum 0) by 0

        self.W = W
        return self

    def _compute_phi(self, x: np.ndarray) -> np.ndarray:
        """
        Compute angle phi_k = arccos(2*x_k - 1) for a feature vector x in [0, 1].
        """
        x_clipped = np.clip(x, 0.0, 1.0)
        z = 2.0 * x_clipped - 1.0
        z = np.clip(z, -1.0, 1.0)
        phi = np.arccos(z)
        return phi

    def _build_iGASF_iGADF(self, x: np.ndarray):
        """
        Build iGASF and iGADF for a single sample.
        Uses feature indices as pseudo-time order.
        """
        phi = self._compute_phi(x)  # shape: (D,)
        phi_i = phi[:, None]        # (D, 1)
        phi_j = phi[None, :]        # (1, D)

        iGASF = np.cos(phi_i + phi_j)     # (D, D)
        iGADF = np.sin(phi_i - phi_j)     # (D, D)
        return iGASF, iGADF

    def _build_iMTF(self, x: np.ndarray):
        """
        Build iMTF for a single sample.
        - Quantize x into states s_k
        - Use global W to fill iMTF[i, j] = W[s_i, s_j]
        """
        if self.W is None:
            raise RuntimeError("Global transition matrix W is not initialized. Call fit() first.")

        x_clipped = np.clip(x, 0.0, 1.0)
        s = (x_clipped * self.Q).astype(int)
        s = np.clip(s, 0, self.Q - 1)  # shape: (D,)

        # iMTF[i, j] = W[s[i], s[j]]
        # We can vectorize this using broadcasting
        s_i = s[:, None]    # (D, 1)
        s_j = s[None, :]    # (1, D)
        iMTF = self.W[s_i, s_j]  # (D, D)
        return iMTF

    def _to_uint8_rgb(self, iGASF: np.ndarray, iGADF: np.ndarray, iMTF: np.ndarray) -> np.ndarray:
        """
        Map iGASF, iGADF (in [-1, 1]) and iMTF (in [0, 1]) to [0, 255] and stack as RGB.
        """
        # Clip just in case of numerical drift
        iGASF_clipped = np.clip(iGASF, -1.0, 1.0)
        iGADF_clipped = np.clip(iGADF, -1.0, 1.0)
        iMTF_clipped = np.clip(iMTF, 0.0, 1.0)

        # Map [-1, 1] -> [0, 255]
        iGASF_rgb = ((iGASF_clipped + 1.0) / 2.0 * 255.0).astype(np.uint8)
        iGADF_rgb = ((iGADF_clipped + 1.0) / 2.0 * 255.0).astype(np.uint8)

        # Map [0, 1] -> [0, 255]
        iMTF_rgb = (iMTF_clipped * 255.0).astype(np.uint8)

        # Stack as RGB (H, W, 3)
        rgb = np.stack([iGADF_rgb, iGASF_rgb, iMTF_rgb], axis=-1)
        return rgb

    def build_and_save_images(
        self,
        df: pd.DataFrame,
        output_dir: str,
        prefix: str = "train"
    ):
        """
        Build SRF RGB images for all rows in df and save them as PNG files.

        File name format:
        {prefix}_idx{idx:06d}_label-{label}.png

        - df: DataFrame with preprocessed features and label as last column.
        - output_dir: directory to store generated images.
        - prefix: "train" or "test" etc.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Extract features and labels (this will also set D etc. if not set yet)
        X, y = self._extract_features_and_labels(df)

        # Loop over all samples
        for idx in range(X.shape[0]):
            x = X[idx]       # 1D array of length D
            label = y[idx]

            # Build SRF components
            iGASF, iGADF = self._build_iGASF_iGADF(x)
            iMTF = self._build_iMTF(x)

            # Convert to uint8 RGB
            rgb = self._to_uint8_rgb(iGASF, iGADF, iMTF)

            # Create PIL image
            img = Image.fromarray(rgb)

            # Safe file name (avoid strange characters in label)
            label_str = str(label).replace("/", "_").replace("\\", "_").replace(" ", "")
            filename = f"{prefix}_idx{idx:06d}_label-{label_str}.png"
            filepath = os.path.join(output_dir, filename)

            img.save(filepath)

        print(f"Saved {X.shape[0]} images to {output_dir}")


def main():
    # Paths to preprocessed CSVs
    train_csv_path = "awid3_train_preprocessed.csv"
    test_csv_path = "awid3_test_preprocessed.csv"

    # Output directories for images
    train_img_dir = "train_images"
    test_img_dir = "test_images"

    print("Loading preprocessed train and test CSV files...")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Fit SRF builder on train (for iMTF's global transition matrix)
    print("Fitting SRFImageBuilder on train set...")
    srf_builder = SRFImageBuilder(Q=16)
    srf_builder.fit(train_df)

    # Build images for train set
    print("Building train images...")
    srf_builder.build_and_save_images(train_df, output_dir=train_img_dir, prefix="train")

    # Build images for test set (reusing the same W and Q)
    print("Building test images...")
    srf_builder.build_and_save_images(test_df, output_dir=test_img_dir, prefix="test")

    print("Done.")


if __name__ == "__main__":
    main()
