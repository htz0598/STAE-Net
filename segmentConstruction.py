import os
import re
import numpy as np
from PIL import Image
import pandas as pd  # only for potential future use / consistency, not strictly needed


def parse_idx_and_label(filename: str):
    """
    Parse idx and label from filenames like:
      'train_idx000123_label-Normal.png'
      'test_idx000045_label-Deauth.csv'
    Returns:
      idx: int
      label: str
    """
    # idx part
    m_idx = re.search(r"idx(\d+)", filename)
    if m_idx:
        idx = int(m_idx.group(1))
    else:
        raise ValueError(f"Cannot parse idx from filename: {filename}")

    # label part
    m_label = re.search(r"label-([^\.]+)", filename)
    if m_label:
        label = m_label.group(1)
    else:
        # fallback: unknown label
        label = "Unknown"

    return idx, label


def load_image_as_array(path: str):
    """
    Load an RGB image as a NumPy array of shape (H, W, 3), dtype uint8.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr


def build_segments_from_image_dir(
    img_dir: str,
    seg_dir: str,
    prefix: str = "train",
    window_size: int = 8
):
    """
    From a directory of single-frame RGB images (train_images or test_images),
    build sliding-window segments and save as .npz files.

    Each segment is a 4D array of shape (T, H, W, 3), where T = window_size.
    File name format:
      {prefix}_seg_endidx{end_idx:06d}_from{start_idx:06d}_to{end_idx:06d}_label-{last_label}.npz

    Inside each .npz:
      - frames: np.ndarray, shape (T, H, W, 3), dtype uint8
      - indices: np.ndarray, shape (T,), original idx for each frame
      - labels: np.ndarray, shape (T,), label for each frame (string)
    """
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir, exist_ok=True)

    # List all PNG images in the directory
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]

    if not files:
        print(f"No PNG files found in {img_dir}, skipping.")
        return

    # Parse idx and sort by idx to preserve original order
    file_info = []
    for fname in files:
        idx, label = parse_idx_and_label(fname)
        file_info.append((idx, label, fname))

    file_info.sort(key=lambda x: x[0])  # sort by idx ascending

    indices = [fi[0] for fi in file_info]
    labels = [fi[1] for fi in file_info]
    paths = [os.path.join(img_dir, fi[2]) for fi in file_info]

    N = len(paths)
    print(f"{prefix}: found {N} images in {img_dir}")

    if N < window_size:
        print(f"{prefix}: not enough images ({N}) for window_size={window_size}, no segments will be created.")
        return

    # Pre-load all images into memory (if memory is a concern, you can load on-the-fly instead)
    print(f"{prefix}: loading all images into memory...")
    images = [load_image_as_array(p) for p in paths]
    H, W, C = images[0].shape
    assert C == 3, "Each image must have 3 channels (RGB)."

    # Sliding window: for end_idx in [window_size-1, N-1]
    seg_count = 0
    for end_pos in range(window_size - 1, N):
        start_pos = end_pos - window_size + 1

        # Collect frames, indices, labels
        frames = np.stack(images[start_pos:end_pos + 1], axis=0)  # shape (T, H, W, 3)
        seg_indices = np.array(indices[start_pos:end_pos + 1], dtype=np.int64)
        seg_labels = np.array(labels[start_pos:end_pos + 1], dtype=object)

        last_idx = seg_indices[-1]
        last_label = seg_labels[-1]

        # Build filename
        last_label_str = str(last_label).replace("/", "_").replace("\\", "_").replace(" ", "")
        fname = (
            f"{prefix}_seg_endidx{last_idx:06d}"
            f"_from{seg_indices[0]:06d}_to{seg_indices[-1]:06d}"
            f"_label-{last_label_str}.npz"
        )
        out_path = os.path.join(seg_dir, fname)

        # Save as compressed npz
        np.savez_compressed(
            out_path,
            frames=frames,
            indices=seg_indices,
            labels=seg_labels
        )

        seg_count += 1

    print(f"{prefix}: created {seg_count} segments in {seg_dir}")


def main():
    # Input: RGB single-frame images generated in the previous step
    train_img_dir = "train_images"
    test_img_dir = "test_images"

    # Output: sliding-window segments
    train_seg_dir = "train_segments"
    test_seg_dir = "test_segments"

    # Window size T (for AWID3, the paper uses T = 8)
    WINDOW_SIZE = 8

    # Build segments for train
    print("Building train segments...")
    build_segments_from_image_dir(
        img_dir=train_img_dir,
        seg_dir=train_seg_dir,
        prefix="train",
        window_size=WINDOW_SIZE
    )

    # Build segments for test
    print("Building test segments...")
    build_segments_from_image_dir(
        img_dir=test_img_dir,
        seg_dir=test_seg_dir,
        prefix="test",
        window_size=WINDOW_SIZE
    )

    print("Done.")


if __name__ == "__main__":
    main()
