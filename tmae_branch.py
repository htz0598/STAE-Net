import os
import re
import csv
import random
import numpy as np

import torch
import torch.nn as nn
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
    label = m_label.group(1) if m_label else "Unknown"
    return end_idx, label


class SegmentDataset(Dataset):
    """
    Loads temporal segments from .npz.
    Each sample:
      frames: (T, 3, H, W) float32 in [0,1]
      end_idx: int
      label: str (label of last frame)
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

        info.sort(key=lambda x: x[0])  # temporal order by end_idx
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
        # frames: (T, H, W, 3) uint8
        frames = data["frames"].astype(np.float32) / 255.0
        # -> (T, 3, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.from_numpy(frames)
        return frames, end_idx, label


# =========================
# TMAE model (paper-aligned)
# =========================
class TMAE(nn.Module):
    """
    Temporal Masked Autoencoder over SRF image sequences.

    Changes vs your version:
    - Learnable 3D positional embedding: pos_t + pos_h + pos_w
    - random_masking returns ids_keep explicitly (no recomputation hacks)
    - Separate encoder/decoder dims with projection enc->dec
    - Pre-norm transformer blocks (norm_first=True) closer to ViT/MAE style
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        T=8,
        tube_size=2,
        in_chans=3,
        enc_embed_dim=512,
        dec_embed_dim=512,
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
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.mask_ratio = mask_ratio

        self.Pt = T // tube_size
        self.Ph = img_size // patch_size
        self.Pw = img_size // patch_size
        self.num_tokens = self.Pt * self.Ph * self.Pw

        # token dim before projection
        self.token_dim = in_chans * tube_size * patch_size * patch_size

        # token -> encoder embedding
        self.proj = nn.Linear(self.token_dim, enc_embed_dim)

        # 3D learnable positional embeddings (encoder)
        self.pos_t_enc = nn.Parameter(torch.zeros(1, self.Pt, enc_embed_dim))
        self.pos_h_enc = nn.Parameter(torch.zeros(1, self.Ph, enc_embed_dim))
        self.pos_w_enc = nn.Parameter(torch.zeros(1, self.Pw, enc_embed_dim))

        # encoder blocks
        enc_layer = nn.TransformerEncoderLayer(
            d_model=enc_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(enc_embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth_encoder)

        # encoder -> decoder projection (MAE style)
        self.enc_to_dec = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)

        # 3D learnable positional embeddings (decoder)
        self.pos_t_dec = nn.Parameter(torch.zeros(1, self.Pt, dec_embed_dim))
        self.pos_h_dec = nn.Parameter(torch.zeros(1, self.Ph, dec_embed_dim))
        self.pos_w_dec = nn.Parameter(torch.zeros(1, self.Pw, dec_embed_dim))

        # learnable mask token in decoder space
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

        # decoder blocks
        dec_layer = nn.TransformerEncoderLayer(
            d_model=dec_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(dec_embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=depth_decoder)

        # decoder -> token prediction
        self.head = nn.Linear(dec_embed_dim, self.token_dim)

        self._init_weights()

    def _init_weights(self):
        # match MAE-ish init vibes
        nn.init.trunc_normal_(self.pos_t_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_h_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_w_enc, std=0.02)

        nn.init.trunc_normal_(self.pos_t_dec, std=0.02)
        nn.init.trunc_normal_(self.pos_h_dec, std=0.02)
        nn.init.trunc_normal_(self.pos_w_dec, std=0.02)

        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

        nn.init.normal_(self.enc_to_dec.weight, std=0.02)
        nn.init.zeros_(self.enc_to_dec.bias)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        nn.init.normal_(self.mask_token, std=0.02)

    # --------- 3D pos embed -> (1, N, D) ----------
    def _build_pos_embed(self, which: str):
        """
        Build flattened positional embedding:
          pos[t,h,w] = pos_t[t] + pos_h[h] + pos_w[w]
        Return shape: (1, N, D)
        """
        if which == "enc":
            pos_t, pos_h, pos_w = self.pos_t_enc, self.pos_h_enc, self.pos_w_enc
        elif which == "dec":
            pos_t, pos_h, pos_w = self.pos_t_dec, self.pos_h_dec, self.pos_w_dec
        else:
            raise ValueError("which must be 'enc' or 'dec'")

        # (1, Pt, D) -> (1, Pt, 1, 1, D)
        pt = pos_t[:, :, None, None, :]
        ph = pos_h[:, None, :, None, :]
        pw = pos_w[:, None, None, :, :]
        pos = pt + ph + pw  # (1, Pt, Ph, Pw, D)
        pos = pos.reshape(1, self.num_tokens, -1)
        return pos

    # --------- patchify / unpatchify ----------
    def frames_to_tokens(self, frames):
        """
        frames: (B, T, 3, H, W)
        tokens: (B, N, token_dim)
        """
        B, T, C, H, W = frames.shape
        assert T == self.T and C == self.in_chans
        assert H == self.img_size and W == self.img_size

        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)

        Tt = self.tube_size
        Pt, Ph, Pw = self.Pt, self.Ph, self.Pw
        P = self.patch_size

        # temporal split: T = Pt * Tt
        x = x.view(B, C, Pt, Tt, H, W)
        # spatial split
        x = x.view(B, C, Pt, Tt, Ph, P, Pw, P)

        # (B, Pt, Ph, Pw, C, Tt, P, P)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()

        tokens = x.view(B, self.num_tokens, self.token_dim)
        return tokens

    # --------- masking ----------
    def random_masking(self, x, mask_ratio):
        """
        Per-sample random masking.
        x: (B, N, D)
        Returns:
          x_vis: (B, N_vis, D)
          mask: (B, N) 0=visible, 1=masked
          ids_keep: (B, N_vis)
          ids_restore: (B, N)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)      # (B, N)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, mask, ids_keep, ids_restore

    def forward(self, frames):
        """
        frames: (B, T, 3, H, W) in [0,1]
        Returns:
          loss: scalar (masked MSE)
          per_sample_error: (B,) masked MSE per sample
        """
        B = frames.size(0)

        # patchify -> tokens
        tokens = self.frames_to_tokens(frames)               # (B, N, token_dim)
        x = self.proj(tokens)                                # (B, N, enc_dim)

        # add encoder 3D pos
        pos_enc = self._build_pos_embed("enc").to(x.dtype).to(x.device)  # (1, N, enc_dim)
        x = x + pos_enc

        # random masking
        x_vis, mask, ids_keep, ids_restore = self.random_masking(x, self.mask_ratio)  # x_vis:(B,Nv,enc_dim)

        # encoder
        enc = self.encoder(x_vis)                            # (B, Nv, enc_dim)

        # map enc -> decoder dim
        dec_vis = self.enc_to_dec(enc)                       # (B, Nv, dec_dim)

        # prepare decoder tokens: visible + mask, then restore order
        N = self.num_tokens
        Nv = dec_vis.size(1)
        Nm = N - Nv

        mask_tokens = self.mask_token.expand(B, Nm, -1)      # (B, Nm, dec_dim)
        dec_ = torch.cat([dec_vis, mask_tokens], dim=1)      # (B, N, dec_dim) but in shuffled order

        # restore to original token order
        dec_ = torch.gather(
            dec_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_embed_dim)
        )                                                    # (B, N, dec_dim)

        # add decoder 3D pos
        pos_dec = self._build_pos_embed("dec").to(dec_.dtype).to(dec_.device)  # (1, N, dec_dim)
        dec_ = dec_ + pos_dec

        # decoder
        dec = self.decoder(dec_)                             # (B, N, dec_dim)

        # predict tokens
        pred_tokens = self.head(dec)                         # (B, N, token_dim)

        # masked MSE loss
        diff2 = (pred_tokens - tokens) ** 2                  # (B, N, token_dim)
        loss_per_token = diff2.mean(dim=-1)                  # (B, N)

        loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        per_sample = (loss_per_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return loss, per_sample


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
        enc_embed_dim=512,
        dec_embed_dim=512,
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
        all_idx, all_labels, all_errors = [], [], []
        with torch.no_grad():
            for frames, end_idx, labels in dataloader:
                frames = frames.to(device)
                _, per_sample_err = model(frames)
                all_idx.extend(end_idx.numpy().tolist())
                all_labels.extend(list(labels))
                all_errors.extend(per_sample_err.cpu().numpy().tolist())
        print(f"{split_name}: computed temporal scores for {len(all_idx)} segments.")
        return all_idx, all_labels, all_errors

    
    print("Computing temporal errors on TRAIN segments...")
    train_loader_eval = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    train_idx, train_labels, train_errors = compute_scores(train_loader_eval, "Train")

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
