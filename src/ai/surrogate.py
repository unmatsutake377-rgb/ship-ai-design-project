"""물리 라벨 대리모델 (spec §7 M5b-1, 2026-07-28).

역할: 45파라미터 → (타당성 확률, 저항, 중량, 안정여유) 근사.
- 학습 라벨은 전부 우리 물리 파이프라인 산출 (spec §2.5 — 물리가 주 신호)
- 용도는 **후보 추천(순위)** — 최종 수치는 항상 실물리 재검증 (virtual_screen)
- 소표본(수백 척) 학습 — 정밀 예측기가 아님을 metrics로 정직하게 보고

구조: 공유 몸통 MLP + 두 머리 (타당성 sigmoid / 목적 3회귀).
회귀 손실은 feasible 표본에만 적용 (infeasible의 NaN 라벨 마스킹).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIDDEN = 64
VAL_FRACTION = 0.2


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class SurrogateModel:
    net: object          # torch.nn.Module
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray   # 목적 3개 정규화 (feasible 기준)
    y_std: np.ndarray

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """반환: (타당성 확률 [n], 목적 예측 [n,3] — 저항·중량·여유)."""
        import torch

        self.net.eval()
        xs = (np.atleast_2d(x) - self.x_mean) / self.x_std
        with torch.no_grad():
            t = torch.tensor(xs, dtype=torch.float32, device=_device())
            feas_logit, obj_norm = self.net(t)
            feas = torch.sigmoid(feas_logit).cpu().numpy().ravel()
            obj = obj_norm.cpu().numpy() * self.y_std + self.y_mean
        return feas, obj


def _build_net(n_in: int = 45):
    """n_in: 입력 차원 — 원시 45 또는 물리 특징 22 (특징 공학)."""
    import torch.nn as nn

    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(n_in, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            )
            self.head_feas = nn.Linear(HIDDEN, 1)
            self.head_obj = nn.Linear(HIDDEN, 3)

        def forward(self, x):
            h = self.body(x)
            return self.head_feas(h), self.head_obj(h)

    return TwoHead()


def train_surrogate(x: np.ndarray, y_feas: np.ndarray, y_obj: np.ndarray,
                    epochs: int = 400, seed: int = 0,
                    lr: float = 1e-3) -> tuple[SurrogateModel, dict]:
    """학습 + 검증 지표. y_obj는 infeasible 행에 NaN 허용 (마스킹됨).

    반환 metrics: {n_train, n_val, feas_accuracy, obj_r2: [3개]}.
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(x)
    order = rng.permutation(n)
    n_val = max(1, int(VAL_FRACTION * n))
    val_idx, train_idx = order[:n_val], order[n_val:]

    x_mean, x_std = x.mean(0), x.std(0) + 1e-9
    feas_mask = np.isfinite(y_obj).all(axis=1)
    y_mean = y_obj[feas_mask].mean(0)
    y_std = y_obj[feas_mask].std(0) + 1e-9

    xs = (x - x_mean) / x_std
    ys = np.where(feas_mask[:, None], (y_obj - y_mean) / y_std, 0.0)

    dev = _device()
    tx = torch.tensor(xs, dtype=torch.float32, device=dev)
    tf = torch.tensor(y_feas.astype(np.float32), device=dev).unsqueeze(1)
    ty = torch.tensor(ys, dtype=torch.float32, device=dev)
    tm = torch.tensor(feas_mask.astype(np.float32), device=dev).unsqueeze(1)
    tr = torch.tensor(train_idx, device=dev)

    net = _build_net(n_in=x.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logit, obj = net(tx[tr])
        loss_f = bce(logit, tf[tr])
        # 회귀 손실은 feasible에만 (마스킹 평균)
        se = ((obj - ty[tr]) ** 2).mean(dim=1, keepdim=True)
        loss_o = (se * tm[tr]).sum() / tm[tr].sum().clamp(min=1.0)
        (loss_f + loss_o).backward()
        opt.step()

    model = SurrogateModel(net=net, x_mean=x_mean, x_std=x_std,
                           y_mean=y_mean, y_std=y_std)

    # 검증 지표
    feas_p, obj_p = model.predict(x[val_idx])
    feas_acc = float(((feas_p > 0.5) == y_feas[val_idx].astype(bool)).mean())
    r2 = []
    vmask = feas_mask[val_idx]
    for k in range(3):
        if vmask.sum() >= 2:
            yt, yp = y_obj[val_idx][vmask, k], obj_p[vmask, k]
            ss_res = float(((yt - yp) ** 2).sum())
            ss_tot = float(((yt - yt.mean()) ** 2).sum()) + 1e-12
            r2.append(1.0 - ss_res / ss_tot)
        else:
            r2.append(float("nan"))
    metrics = {"n_train": int(len(train_idx)), "n_val": int(n_val),
               "feas_accuracy": feas_acc, "obj_r2": r2}
    return model, metrics
