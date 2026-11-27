"""اختبارات إن البرنامج المدموج بيدّي نفس نتايج البرامج التلاتة القديمة."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calib  # noqa: E402

CONC = [0.0, 5.0, 18.1, 41.4, 97.3]
ABS = [0.0000, 0.0104, 0.0400, 0.0798, 0.1565]


# ── BK-280 القديم: حل دقيق 4×4 بعد استبعاد (0,0) ──
def _old_bk280(abs_, conc):
    abs_, conc = np.array(abs_), np.array(conc)
    m = (abs_ == 0) & (conc == 0)
    x, y = abs_[~m], conc[~m]
    A = np.vstack([x ** 3, x ** 2, x, np.ones_like(x)]).T
    return np.linalg.solve(A, y)


def test_cubic_matches_old_bk280_exactly():
    x, y, zero = calib.clean_points(ABS, CONC)
    assert zero and len(x) == 4
    np.testing.assert_allclose(calib.fit_cubic(x, y), _old_bk280(ABS, CONC), rtol=1e-9, atol=1e-9)


def test_cubic_matches_old_my_spline():
    # My-spline كان بيشيل الصفوف اللي الاتنين فيها صفر وبعدين polyfit درجة 3
    x = np.array(ABS[1:]); y = np.array(CONC[1:])
    x2, y2, _ = calib.clean_points(ABS + [0, 0], CONC + [0, 0])   # صفوف فاضية زيادة
    np.testing.assert_allclose(calib.fit_cubic(x2, y2), np.polyfit(x, y, 3))


def test_five_points_without_zero_no_longer_crashes():
    # البرنامج القديم كان بيقع هنا (np.linalg.solve على مصفوفة 5×4)
    x, y, _ = calib.clean_points([0.01, 0.04, 0.08, 0.15, 0.2], [5, 18, 41, 97, 130])
    assert len(calib.fit_cubic(x, y)) == 4


# ── Vit-D القديم: natural spline بيعدّي على (0,0) ──
def _old_vitd(abs_input):
    x = np.array(ABS); y = np.array(CONC)
    a, b, c, d, k = calib.natural_spline(x, y)  # نفس الكود حرفياً منقول
    idx = np.searchsorted(k, abs_input) - 1
    idx = max(0, min(idx, len(k) - 2))
    dx = abs_input - k[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


@pytest.mark.parametrize("v", [0.0, 0.005, 0.0104, 0.03, 0.06, 0.1, 0.1565])
def test_spline_matches_vitd_and_passes_through_points(v):
    x, y, _ = calib.clean_points(ABS, CONC, keep_zero=True)
    assert len(x) == 5   # (0,0) فضلت
    got, _ = calib.predict_spline(calib.natural_spline(x, y), v)
    assert got == pytest.approx(_old_vitd(v))


def test_spline_interpolates_every_standard():
    x, y, _ = calib.clean_points(ABS, CONC, keep_zero=True)
    sp = calib.natural_spline(x, y)
    for xi, yi in zip(x, y):
        assert calib.predict_spline(sp, xi)[0] == pytest.approx(yi, abs=1e-9)


def test_spline_against_scipy_natural():
    scipy = pytest.importorskip("scipy.interpolate")
    x, y, _ = calib.clean_points(ABS, CONC, keep_zero=True)
    cs = scipy.CubicSpline(x, y, bc_type="natural")
    sp = calib.natural_spline(x, y)
    for v in np.linspace(0, 0.1565, 25):
        assert calib.predict_spline(sp, v)[0] == pytest.approx(float(cs(v)), rel=1e-7, abs=1e-7)


# ── أدوات My-spline ──
def test_validation_score_thresholds_unchanged():
    assert calib.validation_score(0.99999, 0.5, 0.9995)[0] == "🟢 Excellent"
    assert calib.validation_score(0.9995, 2.5, 0.996)[0] == "🔵 Good"         # 2+2+1
    assert calib.validation_score(0.995, 2.5, 0.98)[0] == "🟡 Acceptable"     # 1+2+0
    assert calib.validation_score(0.9, 10, 0.9)[0] == "🔴 Failed"


def test_max_residual_ignores_blank_instead_of_returning_zero():
    y = np.array([0.0, 10, 20])
    pred = np.array([0.5, 11, 20])     # 10% انحراف في النقطة التانية
    assert calib.max_residual_pct(y, pred) == pytest.approx(10.0)


def test_outliers_skip_blank():
    x = np.array([0.001, 0.01, 0.04, 0.08, 0.15])
    y = np.array([0.0, 5, 18, 41, 97])
    co = calib.fit_cubic(x, y)
    assert all(i != 0 for i, _ in calib.detect_outliers(x, y, co, 0.0001))


def test_one_point_factor():
    x, y, f, k, cv = calib.one_point_factor([0.1, 0.2, 0.4], [10, 20, 40])
    assert k == pytest.approx(100) and cv == pytest.approx(0, abs=1e-9)


def test_duplicate_absorbance_rejected():
    assert calib.check_points(np.array([0.1, 0.1, 0.2, 0.3]), np.array([1, 2, 3, 4]))


def test_delta_and_range():
    assert calib.delta_pct(10, 12) == pytest.approx(20)
    assert calib.delta_pct(0, 5) == 0
    assert calib.in_range([0.01, 0.15], 0.1) and not calib.in_range([0.01, 0.15], 0.2)
