import os
import re
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Utils: reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset for temporal segments
# =========================
def parse_endidx_and_label_from_seg(filename: str):
    """
    Parse end_idx and label from filenames like:
      'train_seg_endidx000123_from000116_to000123_label-Normal.npz'
    """
    m_end = re.search(r"endidx(\d+)", filename)
    if not m_end:
        raise ValueError(f"Cannot parse endidx from filename: {filename}")
    end_idx = int(m_end.group(1))

    m_label = re.search(r"label-([^\.]+)", filename)
    if m_label:
        label = m_label.group(1)
    else:
        label = "Unknown"

    return end_idx, label


class SegmentDataset(Dataset):
    """
    Dataset that loads temporal segments from .npz files.
    Each sample:
      - frames: (T, 3, H, W) float32 in [0,1]
      - end_idx: int (segment's last packet index)
      - label: str (label of the last frame)
    """

    def __init__(self, seg_dir: str):
        self.seg_dir = seg_dir
        files = [f for f in os.listdir(seg_dir) if f.lower().endswith(".npz")]
        if not files:
            raise RuntimeError(f"No .npz files found in {seg_dir}")

        info = []
        for fname in files:
            end_idx, label = parse_endidx_and_label_from_seg(fname)
            info.append((end_idx, label, fname))

        # sort by end_idx to preserve temporal order
        info.sort(key=lambda x: x[0])
        self.end_indices = [it[0] for it in info]
        self.labels = [it[1] for it in info]
        self.paths = [os.path.join(seg_dir, it[2]) for it in info]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        end_idx = self.end_indices[i]
        label = self.labels[i]

        data = np.load(path)
        # data["frames"]: (T, H, W, 3), uint8
        frames = data["frames"].astype(np.float32) / 255.0
        # to (T, 3, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.from_numpy(frames)  # (T, 3, H, W)

        return frames, end_idx, label


# =========================
# TMAE model
# =========================
class TMAE(nn.Module):
    """
    Temporal Masked Autoencoder over SRF image sequences.

    - Input: frames of shape (B, T, 3, H, W), values in [0,1]
    - Tube tokenization:
        temporal tube size Tt, spatial patch size P
        total tokens N = (T/Tt) * (H/P) * (W/P)
    - Masked autoencoding:
        randomly mask a ratio of tokens, encode only visible tokens,
        decode full tokens, reconstruct masked tokens.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        T=8,
        tube_size=2,
        in_chans=3,
        embed_dim=512,
        depth_encoder=3,
        depth_decoder=3,
        num_heads=8,
        mlp_ratio=4.0,
        mask_ratio=0.35,
    ):
        super().__init__()
        assert T % tube_size == 0, "T must be divisible by tube_size"
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.T = T
        self.tube_size = tube_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.num_tubes_t = T // tube_size
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_tokens = self.num_tubes_t * self.num_patches_h * self.num_patches_w

        # token dimension before projection
        self.token_dim = in_chans * tube_size * patch_size * patch_size

        # linear projection to embedding space
        self.proj = nn.Linear(self.token_dim, embed_dim)

        # positional embedding for tokens (1D for all tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        # encoder: Transformer encoder over visible tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth_encoder)

        # decoder: Transformer over full tokens (visible+masked)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth_decoder)

        # projection back to token space
        self.head = nn.Linear(embed_dim, self.token_dim)

        # learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.mask_token, std=0.02)

    # --------- patchify / unpatchify ----------
    def frames_to_tokens(self, frames):
        """
        frames: (B, T, 3, H, W)
        return:
          tokens: (B, N, token_dim),
          where N = num_tokens.
        """
        B, T, C, H, W = frames.shape
        assert T == self.T
        assert C == self.in_chans
        assert H == self.img_size and W == self.img_size

        # (B, T, 3, H, W) -> (B, 3, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        Tt = self.tube_size
        Pt = self.num_tubes_t
        Ph = self.num_patches_h
        Pw = self.num_patches_w
        P = self.patch_size

        # reshape temporal: T = Pt * Tt
        x = x.view(B, C, Pt, Tt, H, W)  # (B, C, Pt, Tt, H, W)
        # reshape spatial: H = Ph * P, W = Pw * P
        x = x.view(B, C, Pt, Tt, Ph, P, Pw, P)  # (B, C, Pt, Tt, Ph, P, Pw, P)

        # move tube indices to front: (B, Pt, Ph, Pw, C, Tt, P, P)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        # combine Pt, Ph, Pw into N
        x = x.reshape(B, self.num_tokens, self.token_dim)  # (B, N, token_dim)
        return x

    def tokens_to_frames(self, tokens):
        """
        tokens: (B, N, token_dim)
        return:
          frames: (B, T, 3, H, W)
        """
        B, N, D = tokens.shape
        assert N == self.num_tokens
        assert D == self.token_dim

        C = self.in_chans
        Tt = self.tube_size
        Pt = self.num_tubes_t
        Ph = self.num_patches_h
        Pw = self.num_patches_w
        P = self.patch_size
        T = self.T
        H = self.img_size
        W = self.img_size

        # (B, N, token_dim) -> (B, Pt, Ph, Pw, C, Tt, P, P)
        x = tokens.view(B, Pt, Ph, Pw, C, Tt, P, P)
        # (B, C, Pt, Tt, Ph, P, Pw, P)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # (B, C, T, H, W)
        x = x.contiguous().view(B, C, T, H, W)
        # (B, T, C, H, W)
        frames = x.permute(0, 2, 1, 3, 4)
        return frames

    # --------- masking ----------
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking.
        x: (B, N, D)
        Returns:
          x_vis: visible tokens (B, N_vis, D)
          mask: (B, N) with 0 for visible, 1 for masked
          ids_restore: indices to recover original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # smaller is more likely to be kept

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # build mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, mask, ids_restore

    def forward(self, frames):
        """
        frames: (B, T, 3, H, W) in [0,1]
        Returns:
          loss: scalar
          per_sample_error: (B,) anomaly-like reconstruction error
        """
        B = frames.shape[0]

        # patchify
        tokens = self.frames_to_tokens(frames)  # (B, N, token_dim)
        # project to embedding
        emb = self.proj(tokens)  # (B, N, embed_dim)

        # masking
        x_vis, mask, ids_restore = self.random_masking(emb, self.mask_ratio)

        # add pos embed to visible
        pos_embed = self.pos_embed  # (1, N, D)
        pos_vis = torch.gather(
            pos_embed.expand(B, -1, -1),
            dim=1,
            index=torch.argsort(ids_restore, dim=1)[:, :x_vis.size(1)].unsqueeze(-1).expand(-1, -1, emb.size(-1))
        )  # not perfect but okay; simpler: gather matching ids_keep

        # NOTE:
        # For simplicity and stability, we instead gather by ids_keep directly
        # to avoid the awkward unsort above.
        # Recompute ids_keep from ids_restore:
        # ids_shuffle = torch.argsort(ids_restore, dim=1)
        # ids_keep = ids_shuffle[:, :x_vis.size(1)]
        # pos_vis = torch.gather(pos_embed.expand(B, -1, -1), dim=1,
        #                        index=ids_keep.unsqueeze(-1).expand(-1, -1, emb.size(-1)))

        # For clarity, we do it cleanly here:
        Bn, N, D = emb.shape
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        len_keep = x_vis.size(1)
        ids_keep = ids_shuffle[:, :len_keep]
        pos_vis = torch.gather(
            pos_embed.expand(B, -1, -1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        x_vis = x_vis + pos_vis

        # encoder
        enc = self.encoder(x_vis)  # (B, N_vis, D)

        # prepare decoder input: visible tokens + mask tokens, then restore order
        N_total = self.num_tokens
        len_keep = enc.size(1)
        len_mask = N_total - len_keep

        # mask tokens
        mask_tokens = self.mask_token.expand(B, len_mask, -1)

        # concat visible and mask tokens
        x_ = torch.cat([enc, mask_tokens], dim=1)  # (B, N_total, D)
        # unshuffle to original token order
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # add full positional embedding
        x_ = x_ + self.pos_embed

        # decoder
        dec = self.decoder(x_)  # (B, N, D)

        # project back to token space
        pred_tokens = self.head(dec)  # (B, N, token_dim)

        # compute reconstruction loss on masked tokens only
        target_tokens = tokens  # (B, N, token_dim)
        loss_per_token = (pred_tokens - target_tokens) ** 2  # (B, N, token_dim)
        loss_per_token = loss_per_token.mean(dim=-1)  # (B, N)

        # mask: 1 for masked, 0 for visible
        loss_masked_sum = (loss_per_token * mask).sum()
        num_masked = mask.sum()
        loss = loss_masked_sum / (num_masked + 1e-8)

        # per-sample error = average loss over masked tokens for each sample
        loss_per_sample = (loss_per_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B,)

        return loss, loss_per_sample


# =========================
# Training & scoring
# =========================
def train_tmae(
    train_seg_dir="train_segments",
    test_seg_dir="test_segments",
    model_path="tmae_awid3.pth",
    scores_train_csv="tmae_scores_train.csv",
    scores_test_csv="tmae_scores_test.csv",
    batch_size=8,
    num_epochs=60,
    lr=1e-4,
    device=None,
    img_size=256,
    patch_size=16,
    T=8,
    tube_size=2,
    mask_ratio=0.35,
):
    set_seed(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading train segments...")
    train_dataset = SegmentDataset(train_seg_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print("Loading test segments...")
    test_dataset = SegmentDataset(test_seg_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("Building TMAE model...")
    model = TMAE(
        img_size=img_size,
        patch_size=patch_size,
        T=T,
        tube_size=tube_size,
        in_chans=3,
        embed_dim=512,
        depth_encoder=3,
        depth_decoder=3,
        num_heads=8,
        mlp_ratio=4.0,
        mask_ratio=mask_ratio,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------- Training -------------
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        count = 0
        for frames, _, _ in train_loader:
            # frames: (B, T, 3, H, W)
            frames = frames.to(device)

            optimizer.zero_grad()
            loss, _ = model(frames)
            loss.backward()
            optimizer.step()

            bs = frames.size(0)
            running_loss += loss.item() * bs
            count += bs

        epoch_loss = running_loss / max(count, 1)
        print(f"Epoch [{epoch}/{num_epochs}] - Train masked MSE loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved TMAE model to {model_path}")

    # ------------- Scoring -------------
    def compute_scores(dataloader, split_name):
        model.eval()
        all_idx = []
        all_labels = []
        all_errors = []

        with torch.no_grad():
            for frames, end_idx, labels in dataloader:
                frames = frames.to(device)
                loss, per_sample_err = model(frames)  # per_sample_err shape: (B,)
                all_idx.extend(end_idx.numpy().tolist())
                all_labels.extend(list(labels))
                all_errors.extend(per_sample_err.cpu().numpy().tolist())

        print(f"{split_name}: computed temporal scores for {len(all_idx)} segments.")
        return all_idx, all_labels, all_errors

    print("Computing temporal errors on TRAIN segments...")
    train_idx, train_labels, train_errors = compute_scores(train_loader, "Train")

    print("Computing temporal errors on TEST segments...")
    test_idx, test_labels, test_errors = compute_scores(test_loader, "Test")

    # ------------- Save CSV -------------
    def save_scores(path, indices, labels, errors):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "label", "E_seq"])
            for i, lab, e in zip(indices, labels, errors):
                writer.writerow([i, lab, e])
        print(f"Saved scores to: {path}")

    save_scores(scores_train_csv, train_idx, train_labels, train_errors)
    save_scores(scores_test_csv, test_idx, test_labels, test_errors)


def main():
    train_tmae(
        train_seg_dir="train_segments",
        test_seg_dir="test_segments",
        model_path="tmae_awid3.pth",
        scores_train_csv="tmae_scores_train.csv",
        scores_test_csv="tmae_scores_test.csv",
        batch_size=8,
        num_epochs=60,
        lr=1e-4,
        img_size=256,
        patch_size=16,
        T=8,
        tube_size=2,
        mask_ratio=0.35,
    )


if __name__ == "__main__":
    main()
