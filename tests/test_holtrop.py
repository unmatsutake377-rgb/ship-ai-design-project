"""Holtrop-Mennen 검증 (전 크기 개방 1단계, 스펙 2026-08-06).

앵커 2종: ① 부속 공식 손계산 (독립 재계산) ② KCS 공개 벤치마크
(Kim, Van & Kim 2001 — Ct 3.556e-3 @ Fn 0.26, Re 1.4e7, A급).
"""
import math

import pytest

from src.physics.holtrop import (
    HoltropInput,
    correlation_allowance,
    form_factor,
    half_entrance_angle,
    run_length,
    total_resistance_holtrop,
)

# KCS 모형 (Tokyo 2015 EFD 기하 A급 + 워크숍 통상 계수 B급 +
# 구상선수 개략 C급 — ABT는 컨테이너선 통상비 ABT/AM≈9%)
KCS = HoltropInput(lwl=7.358, beam=1.019, draft=0.342, volume=1.649,
                   wetted_surface=9.553, cb=0.650, cm=0.9849, cwp=0.8204,
                   lcb_frac=-0.0148, abt=0.030, hb=0.126)
KCS_V = 0.26 * math.sqrt(9.81 * 7.358)   # Fn 0.26
KCS_CT_EFD = 3.556e-3                    # Kim et al. 2001 (담수 수조)


def test_run_length_hand_calc():
    """LR 손계산: Cp=0.66, lcb=-1.48% →
    LR/L = 1-0.66+0.06·0.66·(-1.48)/(4·0.66-1) = 0.30427."""
    lr = run_length(KCS)
    cp = 0.650 / 0.9849
    expected = 7.358 * (1 - cp + 0.06 * cp * (-1.48) / (4 * cp - 1))
    assert lr == pytest.approx(expected, rel=1e-9)
    assert lr == pytest.approx(2.239, abs=0.005)


def test_form_factor_plausible_range():
    """1+k1: 상선 통상 1.1~1.35 + KCS 실측 계보 (~1.1~1.2)."""
    k1 = form_factor(KCS)
    assert 1.10 < k1 < 1.25


def test_entrance_angle_container_range():
    """iE 회귀: 날씬 컨테이너선 수 도° ~ 20° 대역 (발산 방지)."""
    ie = half_entrance_angle(KCS)
    assert 5.0 < ie < 25.0


def test_kcs_benchmark_overprediction_recorded():
    """KCS 공개 실측 대조 — 실측 박제 (2026-08-06):

    예측 Ct = 4.04e-3 vs EFD 3.556e-3 → +13.6% 과대.
    Holtrop의 fine 컨테이너선(Cp 0.66 = 회귀 DB 하단) 과대 경향
    (문헌 보고 10~15%)과 일치 — 구상선수 파라미터가 개략(C급)인
    한계 포함. 회귀 방지 경계 [1.05, 1.25]: 더 나빠지면 실패,
    극적으로 좋아져도 (공식 변경 의심) 실패."""
    r = total_resistance_holtrop(KCS, KCS_V, rho=998.5, nu=1.139e-6,
                                 include_ca=False)
    ratio = r["ct"] / KCS_CT_EFD
    assert 1.05 < ratio < 1.25
    # 성분 건전성: 점성이 지배 (배수량 저Fn 정상 구조), 조파 > 0
    assert r["rv"] > r["rw"] > 0
    assert r["ra"] == 0.0            # 모형 대조는 CA 제외


def test_correlation_allowance_full_scale_positive():
    """CA: 실선 스케일(230 m급)에서 양수 소량 (통상 2~6e-4)."""
    full = HoltropInput(lwl=232.5, beam=32.2, draft=10.8, volume=52030.0,
                        wetted_surface=9530.0, cb=0.650, cm=0.9849,
                        cwp=0.8204, lcb_frac=-0.0148)
    ca = correlation_allowance(full)
    assert 1e-4 < ca < 8e-4


def test_full_scale_kcs_sane_power():
    """실선 KCS 24 kn: 유효동력 자릿수 검증 (문헌 ~15~25 MW 대역
    — PE = R·V. 공개 설계 계보와 자릿수 일치가 목적)."""
    full = HoltropInput(lwl=232.5, beam=32.2, draft=10.8, volume=52030.0,
                        wetted_surface=9530.0, cb=0.650, cm=0.9849,
                        cwp=0.8204, lcb_frac=-0.0148, abt=30.0, hb=4.0)
    v = 24 * 0.5144
    r = total_resistance_holtrop(full, v, include_ca=True)
    pe_mw = r["total"] * v / 1e6
    assert 10.0 < pe_mw < 30.0
