"""Michell 보정 계수 (스펙 §4) — 능동 학습 1회전의 '학습' 부분.

모형: ratio_w(B/L) = 1 + b·(B/L). 절편은 물리 앵커(B/L→0에서 Michell
정확 → ratio=1)로 고정 — 소표본(4점)에서 매개변수는 b 하나만 배운다.
앵커드 최소제곱: b = Σ(ratio_i − 1)·x_i / Σ x_i²  (x = B/L).

CFD 조파 추정 = P_자유수면 − P_이중모형: 같은 척을 두 모드로 돌려
형상(점성 압력) 성분을 빼내면 파도 몫만 남는다.
"""
from __future__ import annotations

import pandas as pd

RATIO_CLIP = (0.05, 1.5)   # 외삽 폭주 방지 (스펙 §4)


def fit_wave_ratio(points: list[tuple[float, float]]) -> float:
    num = sum((r - 1.0) * x for x, r in points)
    den = sum(x * x for x, _ in points)
    return num / den


def wave_ratio(bl: float, b: float,
               lo: float = RATIO_CLIP[0], hi: float = RATIO_CLIP[1]) -> float:
    return min(hi, max(lo, 1.0 + b * bl))


def ratios_from_labels(df: pd.DataFrame) -> list[tuple[float, float]]:
    """라벨 CSV에서 (B/L, ratio) 점 추출 — _simple_/_inter_ 짝 필수."""
    points = []
    df = df.dropna(subset=["loa_m", "beam_m"])
    bases: dict[str, dict] = {}
    for _, row in df.iterrows():
        if "_simple_" in row.case_name:
            base = row.case_name.replace("_simple_", "_")
            bases.setdefault(base, {})["simple"] = row
        elif "_inter_" in row.case_name:
            base = row.case_name.replace("_inter_", "_")
            bases.setdefault(base, {})["inter"] = row
    for base, pair in sorted(bases.items()):
        if "simple" not in pair or "inter" not in pair:
            continue
        s, i = pair["simple"], pair["inter"]
        ratio = (i.cfd_pressure_n - s.cfd_pressure_n) / i.emp_rw_n
        points.append((float(i.beam_m / i.loa_m), float(ratio)))
    return points


def _mark_pareto(f):
    """3목적 (저항↓, 중량↓, −안정여유↓) 비지배 표시 — screen_shipd와 동일 논리."""
    import numpy as np

    n = len(f)
    flags = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (f[j] <= f[i]).all() and (f[j] < f[i]).any():
                flags[i] = False
                break
    return flags


def reevaluate_pareto(pareto_csv, b: float, speed: float = 1.2):
    """보정 경험식으로 파레토 후보 재평가 (스펙 §5 — Fn 외삽 시연).

    pareto.csv에는 rw/rf 분해가 없으므로 치수(loa, lb, bt, cb)로 선형을
    재생성해 설계 흘수에서 rw·rf를 재계산한다 (평형 흘수가 아니라 설계
    흘수 — 근사임을 리포트에 명시)."""
    import numpy as np

    from src.ai.hull_generator import generate_hull_mesh, solve_exponents
    from src.core.types import MainDimensions
    from src.physics.resistance import total_resistance

    df = pd.read_csv(pareto_csv)
    df = df[df["feasible"]].reset_index(drop=True)
    rows = []
    for _, c in df.iterrows():
        beam = c.loa / c.lb
        draft = beam / c.bt
        dims = MainDimensions(loa=c.loa, beam=beam, depth=2 * draft,
                              draft_design=draft, cb=c.cb)
        n_exp, m_exp = solve_exponents(c.cb)
        r = total_resistance(generate_hull_mesh(dims), dims, n_exp, m_exp,
                             draft=draft, speed=speed)
        ratio = wave_ratio(beam / c.loa, b)
        rows.append({**c, "rw_orig": r.rw, "rf_orig": r.rf, "ratio": ratio,
                     "resistance_corrected": r.rf + r.rw * ratio})
    out = pd.DataFrame(rows)
    before = np.column_stack([out.resistance_n, out.total_mass_kg,
                              -out.stability_margin])
    after = np.column_stack([out.resistance_corrected, out.total_mass_kg,
                             -out.stability_margin])
    out["pareto_before"] = _mark_pareto(before)
    out["pareto_after"] = _mark_pareto(after)
    return out


def plot_reeval(df, b: float, points, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    xs = np.linspace(0, 0.55, 100)
    ax1.plot(xs, [wave_ratio(x, b) for x in xs], "-",
             color="#0f766e", label=f"적합: 1 + ({b:.2f})·(B/L)")
    px, py = zip(*points)
    ax1.scatter(px, py, s=70, color="#dc2626", zorder=5, label="CFD 관측점")
    ax1.set_xlabel("B/L (통통함)")
    ax1.set_ylabel("ratio_w = CFD 조파 / Michell")
    ax1.set_title("Michell 보정 계수 (Fn≈0.34)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.scatter(df.resistance_n, df.total_mass_kg, s=25, alpha=0.4,
                color="#94a3b8", label="보정 전")
    ax2.scatter(df.resistance_corrected, df.total_mass_kg, s=25, alpha=0.7,
                color="#0f766e", label="보정 후")
    fb = df[df.pareto_after]
    ax2.scatter(fb.resistance_corrected, fb.total_mass_kg, s=90,
                facecolor="none", edgecolor="#dc2626", label="보정 후 전선")
    ax2.set_xlabel("전저항 [N] (Fn 외삽 시연)")
    ax2.set_ylabel("전체 중량 [kg]")
    ax2.set_title("파레토 재평가")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="능동 학습 1회전: 적합+재평가")
    parser.add_argument("--labels", default="data/cfd_labels.csv")
    parser.add_argument("--pareto", default="outputs/pareto/pareto.csv")
    parser.add_argument("--out", default="outputs/active_learning")
    args = parser.parse_args()

    points = ratios_from_labels(pd.read_csv(args.labels))
    if len(points) < 2:
        print(f"짝 라벨 부족 ({len(points)}점) — CFD 실행 먼저")
        return 2
    b = fit_wave_ratio(points)
    print(f"적합 b = {b:.3f}  (점 {len(points)}개)")
    for x, r in points:
        print(f"  B/L={x:.3f}: 관측 {r:.3f} vs 적합 {wave_ratio(x, b):.3f}")
    df = reevaluate_pareto(args.pareto, b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "reeval.csv", index=False)
    plot_reeval(df, b, points, out / "reeval.png")
    changed = df[df.pareto_before != df.pareto_after]
    print(f"전선 변동: {len(changed)}척 (전 {df.pareto_before.sum()} → "
          f"후 {df.pareto_after.sum()})")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
