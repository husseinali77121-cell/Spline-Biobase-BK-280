"""calib.py — حسابات المعايرة (بدون Streamlit عشان تتختبر لوحدها).

مدموج من 3 برامج:
  • Spline-Biobase-BK-280  : معاملات Par A–D زي شاشة الجهاز (نقطة الصفر مستبعدة)
  • My-spline-new          : Validation score, outliers, one-point factor, delta check
  • Vit-D_concentraion     : Natural cubic spline (قطع بين كل نقطتين)

ملاحظة مهمة: مع 4 نقاط بعد استبعاد الصفر، polyfit (least squares) بيدّي
نفس الحل الدقيق اللي كان BK-280 بيحسبه بـ np.linalg.solve بالظبط.
مع أكتر من 4 نقاط، polyfit بيشتغل عادي بدل ما البرنامج القديم كان يقف بخطأ.
"""
from __future__ import annotations

import numpy as np

DRIFT_THRESHOLD = 15.0     # % — Delta check
ONE_POINT_CV = 5.0         # % — One point end


# ───────────────────────── تجهيز النقاط ─────────────────────────
def clean_points(abs_vals, conc_vals, keep_zero=False):
    """يرجّع (x, y, zero_found) بعد شيل الصفوف الفاضية، مترتبة بالامتصاصية.

    keep_zero=False (Cubic Par A–D): نقطة (0,0) بتتشال من التركيب زي ما
    الجهاز بيعمل (BK-280 و My-spline الاتنين كانوا بيعملوا كده).
    keep_zero=True (Natural spline): (0,0) بتفضل نقطة في المنحنى زي Vit-D.
    أي Blank بامتصاصية ≠ 0 بيفضل في الحالتين.
    """
    x = np.asarray(abs_vals, dtype=float)
    y = np.asarray(conc_vals, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    zero = (x == 0) & (y == 0)
    if keep_zero and zero.sum() > 1:          # صفوف صفر زيادة (جدول فاضي)
        first = np.argmax(zero)
        drop = zero.copy()
        drop[first] = False
        x, y = x[~drop], y[~drop]
    elif not keep_zero:
        x, y = x[~zero], y[~zero]
    order = np.argsort(x, kind="stable")
    return x[order], y[order], bool(zero.any())


def check_points(x, y):
    """رسالة خطأ بالعربي أو None لو النقاط سليمة."""
    if len(x) < 4:
        return "الرجاء إدخال 4 نقاط على الأقل (بعد استبعاد نقطة الصفر)."
    if len(np.unique(x)) != len(x):
        return "فيه قيمتين امتصاصية متكررين — كل Standard لازم يكون له امتصاصية مختلفة."
    return None


# ───────────────────── Cubic polynomial (Par A–D) ─────────────────────
def fit_cubic(x, y):
    """Par A, B, C, D:  Conc = A·Abs³ + B·Abs² + C·Abs + D"""
    return np.polyfit(np.asarray(x, float), np.asarray(y, float), 3)


def predict_cubic(coeffs, abs_value):
    return np.polyval(coeffs, abs_value)


# ───────────────────── Natural cubic spline (Vit-D) ─────────────────────
def natural_spline(x, y):
    """نفس خوارزمية Vit-D_concentraion حرفياً. يرجّع (a, b, c, d, knots)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x) - 1
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])
    l, mu, z = np.ones(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    a = y
    b, c, d = np.zeros(n), np.zeros(n + 1), np.zeros(n)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    return a[:-1], b, c[:-1], d, x


def predict_spline(spline, abs_value):
    """يرجّع (التركيز, رقم القطعة) — نفس منطق Vit-D."""
    a, b, c, d, knots = spline
    idx = int(np.searchsorted(knots, abs_value) - 1)
    idx = max(0, min(idx, len(knots) - 2))
    dx = abs_value - knots[idx]
    return float(a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3), idx


# ───────────────────────── جودة المعايرة ─────────────────────────
def compute_r2(y_actual, y_predicted):
    y_actual = np.asarray(y_actual, float)
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def residual_pct(y, y_pred):
    """نسبة الانحراف لكل نقطة؛ NaN للنقاط اللي تركيزها صفر (Blank)."""
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    nz = y != 0
    out[nz] = (y[nz] - np.asarray(y_pred)[nz]) / y[nz] * 100
    return out


def max_residual_pct(y, y_pred):
    """أقصى انحراف % على النقاط غير الصفرية.

    FIX: البرنامج القديم كان يرجّع 0 لو فيه أي نقطة تركيزها صفر،
    فالمعايرة كانت تطلع "Excellent" غلط.
    """
    r = residual_pct(y, y_pred)
    r = r[~np.isnan(r)]
    return float(np.max(np.abs(r))) if r.size else 0.0


def linearity_r2(x, y):
    lin = np.polyfit(x, y, 1)
    return compute_r2(y, np.polyval(lin, x))


def validation_score(r2, max_res_pct, r2_lin):
    """(grade, color, description) — نفس معايير My-spline-new."""
    score = 0
    if r2 >= 0.9999:
        score += 3
    elif r2 >= 0.999:
        score += 2
    elif r2 >= 0.99:
        score += 1
    if max_res_pct <= 1:
        score += 3
    elif max_res_pct <= 3:
        score += 2
    elif max_res_pct <= 5:
        score += 1
    if r2_lin >= 0.999:
        score += 2
    elif r2_lin >= 0.995:
        score += 1
    if score >= 7:
        return "🟢 Excellent", "#28a745", "معايرة ممتازة — جاهزة للاستخدام"
    if score >= 5:
        return "🔵 Good", "#007bff", "معايرة جيدة — مقبولة للاستخدام"
    if score >= 3:
        return "🟡 Acceptable", "#ffc107", "معايرة مقبولة — يُنصح بإعادة التحقق"
    return "🔴 Failed", "#dc3545", "معايرة فاشلة — أعد المعايرة"


def detect_outliers(x, y, coeffs, threshold_pct=15):
    """[(index, deviation%)] — نقاط الـ Blank (تركيز صفر) مستبعدة."""
    y_pred = np.polyval(coeffs, x)
    out = []
    for i in range(len(y)):
        if y[i] == 0 or y_pred[i] == 0:
            continue
        pct = abs((y[i] - y_pred[i]) / y_pred[i]) * 100
        if pct > threshold_pct:
            out.append((i, float(pct)))
    return out


def one_point_factor(x, y):
    """(factors, mean_K, cv%) على النقاط الموجبة فقط، أو None."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return None
    f = y[m] / x[m]
    mean = float(np.mean(f))
    cv = float(np.std(f) / mean * 100) if mean else float("nan")
    return x[m], y[m], f, mean, cv


def delta_pct(old, new):
    return 0.0 if old == 0 else float((new - old) / abs(old) * 100)


def in_range(x, abs_value):
    return float(np.min(x)) <= abs_value <= float(np.max(x))
