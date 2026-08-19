# -*- coding: utf-8 -*-
"""
LLM-SC fiziksel katmanı — third_party/LLM_com/test.ipynb'deki akışın temiz portu.
Birebir korunan protokol: token → sabit genişlikte bit → 8QAM → kanal.
Genelleştirme (belgeli): bit genişliği sözlük boyutundan türetilir —
N_bits = 3·ceil(ceil(log2(V))/3)  (8QAM = 3 bit/sembol; Vicuna 32k→15 bit/5 sembol,
Qwen ~151k→18 bit/6 sembol). Kanal olabilirliği: log p(y|token) = -Σ|y - h·c|²/N0.
"""
import math

import numpy as np
import torch

# test.ipynb'deki 8QAM yıldız kümesi (birebir)
_CONSTELLATION_MAP = {
    (0, 0, 0): 1 + 1j,
    (0, 0, 1): -1 + 1j,
    (0, 1, 0): -1 - 1j,
    (0, 1, 1): 1 - 1j,
    (1, 0, 0): (1 + math.sqrt(3)) + 0j,
    (1, 0, 1): 0 + 1j * (1 + math.sqrt(3)),
    (1, 1, 0): 0 - 1j * (1 + math.sqrt(3)),
    (1, 1, 1): (-1 - math.sqrt(3)) + 0j,
}


def bits_per_token(vocab_size: int) -> int:
    raw = math.ceil(math.log2(vocab_size))
    return 3 * math.ceil(raw / 3)


def symbols_per_token(vocab_size: int) -> int:
    return bits_per_token(vocab_size) // 3


def build_constellation(vocab_size: int, device: str = "cuda") -> torch.Tensor:
    """Her token id'si için sembol dizisi: [V, S] complex64.
    (V×S tablo; Qwen-0.8B için ~152k×6×8B ≈ 7 MB — GPU'da tutulur.)"""
    n_bits = bits_per_token(vocab_size)
    n_sym = n_bits // 3
    ids = np.arange(vocab_size, dtype=np.int64)
    # bit matrisi [V, n_bits] (MSB önce — dec2bitarray uyumlu)
    bits = ((ids[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1).astype(np.int8)
    const = np.empty((vocab_size, n_sym), dtype=np.complex64)
    lut = np.empty(8, dtype=np.complex64)
    for b, c in _CONSTELLATION_MAP.items():
        lut[b[0] * 4 + b[1] * 2 + b[2]] = c
    tri = bits.reshape(vocab_size, n_sym, 3)
    idx = tri[:, :, 0] * 4 + tri[:, :, 1] * 2 + tri[:, :, 2]
    const[:] = lut[idx]
    return torch.from_numpy(const).to(device)


class Channel:
    """AWGN / Rayleigh (sembol başına bağımsız h, perfect CSI — test.ipynb protokolü).
    SNR, iletilen bloğun ölçülen ortalama sembol gücüne göre uygulanır."""

    def __init__(self, snr_db: float, kind: str = "AWGN", device: str = "cuda",
                 generator: torch.Generator | None = None):
        self.snr_db = snr_db
        self.kind = kind
        self.device = device
        self.gen = generator

    def _noise(self, shape, sigma2):
        std = math.sqrt(sigma2 / 2.0)
        nr = torch.randn(shape, device=self.device, generator=self.gen) * std
        ni = torch.randn(shape, device=self.device, generator=self.gen) * std
        return torch.complex(nr, ni)

    def transmit(self, tx: torch.Tensor):
        """tx: [N] complex64 sembol dizisi → (y, h, n0)."""
        p_sig = tx.abs().pow(2).mean().item()
        n0 = p_sig / (10 ** (self.snr_db / 10.0))
        if self.kind == "AWGN":
            h = torch.ones_like(tx)
        elif self.kind == "Rayleigh":
            hr = torch.randn(tx.shape, device=self.device, generator=self.gen) * math.sqrt(0.5)
            hi = torch.randn(tx.shape, device=self.device, generator=self.gen) * math.sqrt(0.5)
            h = torch.complex(hr, hi)
        else:
            raise ValueError("kanal: AWGN | Rayleigh")
        y = h * tx + self._noise(tx.shape, n0)
        return y, h, n0


def channel_loglik(y_step: torch.Tensor, h_step: torch.Tensor, n0: float,
                   constellation: torch.Tensor) -> torch.Tensor:
    """Bir token adımının S sembolü için tüm sözlüğün log-olabilirliği: [V].
    log p(y|token) = -Σ_s |y_s - h_s·c_{token,s}|² / N0   (sabit terimler atılır).
    Beam'den bağımsızdır (kanal gözlemi aynı) → adım başına 1 kez hesaplanır."""
    diff = y_step.unsqueeze(0) - h_step.unsqueeze(0) * constellation  # [V, S]
    return -(diff.real.pow(2) + diff.imag.pow(2)).sum(dim=1) / n0
