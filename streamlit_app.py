"""Orange Calibration Tool — BK-280 Spline (مدموج)

دمج 3 برامج في برنامج واحد:
  • Spline-Biobase-BK-280  → Par A–D زي شاشة الجهاز
  • My-spline-new          → Validation, Outliers, One-point, Delta check, Archive, Simulator
  • Vit-D_concentraion     → Natural cubic spline (اختياري)
الحسابات كلها في calib.py ومختبرة في tests/test_calib.py.
"""
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import calib  # noqa: E402

APP_TITLE = "🧪 Orange Calibration Tool — BK-280 Spline"
M_CUBIC = "Cubic — Par A·B·C·D (جهاز BK-280)"
M_SPLINE = "Natural spline — قطع بين النقاط (Vit-D)"

st.set_page_config(page_title="Orange Calibration Tool", page_icon="🧪", layout="wide")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        [data-testid="manage-app-button"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)


def _secret(section, key, default=None):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════ الدخول (اختياري) ══════════════════════
# بيشتغل بس لو الـ Secrets فيها:
#   [allowed_emails]  emails = ["..."]
#   [app]             password = "..."
# من غيرهم البرنامج بيفتح على طول (زي BK-280 القديم).
ALLOWED = _secret("allowed_emails", "emails")
APP_PASS = _secret("app", "password")
LOGIN_ON = bool(ALLOWED) and bool(APP_PASS)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = not LOGIN_ON


def contact_block():
    name = _secret("contact", "name", "Dr / Hussein Ali")
    lines = [f"<p style='margin:4px 0; font-weight:bold;'>{name}</p>"]
    phone = _secret("contact", "phone")
    mail = _secret("contact", "email")
    if phone:
        lines.append(f"<p style='margin:4px 0;'>📱 T: {phone}</p>")
    if mail:
        lines.append(f"<p style='margin:4px 0;'>📧 Email: {mail}</p>")
    st.markdown("<div style='text-align:center; padding:10px 0 20px 0;'>"
                "<p style='color:#E07B00; font-weight:bold; font-size:16px;'>📞 للتواصل &nbsp; Contact</p>"
                + "".join(lines) + "</div>", unsafe_allow_html=True)


if not st.session_state.authenticated:
    st.markdown("<br>", unsafe_allow_html=True)
    col_center = st.columns([1, 1.2, 1])[1]
    with col_center:
        st.markdown(f"## {APP_TITLE}")
        st.markdown("#### 🔐 تسجيل الدخول")
        login_email = st.text_input("البريد الإلكتروني", placeholder="example@lab.com", key="login_email")
        login_pass = st.text_input("كلمة المرور", type="password", key="login_pass")
        if st.button("دخول ▶", type="primary", use_container_width=True):
            ok_mail = login_email.strip().lower() in [str(e).lower() for e in ALLOWED]
            if ok_mail and login_pass == str(APP_PASS):
                st.session_state.authenticated = True
                st.session_state.logged_email = login_email.strip().lower()
                st.rerun()
            else:
                st.error("❌ البريد الإلكتروني أو كلمة المرور غير صحيحة.")
        st.markdown("---")
        contact_block()
    st.stop()

# ══════════════════════ الهيدر ══════════════════════
col_title, col_user = st.columns([4, 1])
with col_title:
    st.title(APP_TITLE)
with col_user:
    if LOGIN_ON:
        st.markdown(f"<br><small>👤 {st.session_state.get('logged_email', '')}</small>",
                    unsafe_allow_html=True)
        if st.button("تسجيل خروج", key="logout"):
            st.session_state.authenticated = False
            st.rerun()

# ══════════════════════ الحالة ══════════════════════
DEFAULT_DF = pd.DataFrame({
    "Absorbance": [0.0000, 0.0104, 0.0400, 0.0798, 0.1565],
    "Concentration": [0.0, 5.0, 18.1, 41.4, 97.3],
})
for k, v in {"df": DEFAULT_DF, "result": None, "prev_coeffs": None, "archive": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════ الإدخال ══════════════════════
method = st.radio("طريقة الحساب", [M_CUBIC, M_SPLINE], horizontal=True, key="method",
                  help="Cubic: المعاملات اللي بتتكتب في الجهاز. Spline: منحنى بيعدّي على كل نقطة.")

st.write("### 📋 جدول بيانات المعايرة")
col_table, col_hint = st.columns([3, 1])
with col_hint:
    if method == M_CUBIC:
        st.info("أدخل الـ Standards من أقل لأعلى تركيز.\n\nنقطة (0,0) بتتشال من التركيب زي ما الجهاز بيعمل.")
    else:
        st.info("أدخل الـ Standards.\n\nفي الـ Spline نقطة (0,0) بتفضل جزء من المنحنى.")
with col_table:
    edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True, key="editor")

cal_name = st.text_input("اسم هذه المعايرة (اختياري):", placeholder="مثال: Ferritin Run#1 — 2026-11-28")

if st.button("🔬 احسب (Calculate)", type="primary"):
    df_in = edited_df.apply(pd.to_numeric, errors="coerce")
    keep_zero = method == M_SPLINE
    x, y, zero = calib.clean_points(df_in["Absorbance"], df_in["Concentration"], keep_zero=keep_zero)
    if keep_zero:
        err = (None if len(x) >= 3 and len(np.unique(x)) == len(x)
               else "الـ Spline محتاج 3 نقاط على الأقل بامتصاصيات مختلفة.")
    else:
        err = calib.check_points(x, y)
    if err:
        st.error(err)
    else:
        st.session_state.df = edited_df
        if method == M_CUBIC:
            coeffs = calib.fit_cubic(x, y)
            old = st.session_state.result
            if old and old.get("method") == M_CUBIC:
                st.session_state.prev_coeffs = old["coeffs"]
            st.session_state.result = dict(method=M_CUBIC, x=x, y=y, coeffs=coeffs, zero=zero)
        else:
            st.session_state.result = dict(method=M_SPLINE, x=x, y=y,
                                           spline=calib.natural_spline(x, y), zero=zero)

res = st.session_state.result
if res is not None and res["method"] != method:
    st.info("غيّرت طريقة الحساب — اضغط «احسب» تاني.")
    res = None


def curve_plot(x, y, xs, ys, title, extra=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, color="blue", linewidth=2, label="Fit")
    ax.scatter(x, y, color="red", zorder=5, s=60, label="Calibrators")
    if extra:
        extra(ax)
    ax.set_xlabel("Absorbance")
    ax.set_ylabel("Concentration")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def simulator(predict, x):
    """محاكي عينة واحدة + Batch — مشترك بين الطريقتين."""
    st.write("### 🔍 Sample Simulator")
    test_abs = st.number_input("أدخل قيمة Absorbance:", min_value=0.0, format="%.4f",
                               step=0.0001, key="test_absorbance")
    if test_abs > 0:
        conc = predict(test_abs)
        st.metric("Concentration المحسوب", f"{conc:.3f}")
        if calib.in_range(x, test_abs):
            st.success(f"✅ القيمة ضمن نطاق المعايرة ({min(x):.4f} → {max(x):.4f})")
        else:
            st.warning(f"⚠️ القيمة {test_abs:.4f} خارج نطاق المعايرة ({min(x):.4f} → {max(x):.4f}) "
                       "— النتيجة تقريبية (Extrapolation)؛ يُنصح بالتخفيف وإعادة القياس.")
    st.write("---")
    st.write("#### 📋 Batch — عدة عينات مرة واحدة:")
    batch = st.text_area("قيم Absorbance (كل قيمة في سطر):", placeholder="0.045\n0.081\n0.120", height=140)
    if st.button("احسب Batch", key="batch_calc"):
        try:
            vals = [float(v.strip()) for v in batch.strip().splitlines() if v.strip()]
            bdf = pd.DataFrame({
                "Sample #": [f"S{i + 1}" for i in range(len(vals))],
                "Absorbance": vals,
                "Concentration": [round(predict(a), 3) for a in vals],
                "Status": ["✅ In Range" if calib.in_range(x, a) else "⚠️ Extrapolation" for a in vals],
            })
            st.dataframe(bdf, use_container_width=True, hide_index=True)
            st.download_button("⬇️ تحميل النتائج CSV", bdf.to_csv(index=False).encode("utf-8"),
                               "batch_results.csv", "text/csv")
        except ValueError as e:
            st.error(f"خطأ في الإدخال: {e}")


# ══════════════════════ نتائج Natural spline ══════════════════════
if res is not None and res["method"] == M_SPLINE:
    x, y, sp = res["x"], res["y"], res["spline"]
    t1, t2 = st.tabs(["📊 النتائج", "🔍 Sample Simulator"])
    with t1:
        st.info("المنحنى بيعدّي على كل Standard بالظبط (R² = 1 دايمًا). "
                "مفيش Par A–D واحدة تتكتب في الجهاز — لكل قطعة معادلتها.")
        a, b, c, d, k = sp
        seg = pd.DataFrame({
            "من Abs": np.round(k[:-1], 4), "إلى Abs": np.round(k[1:], 4),
            "A": np.round(a, 4), "B": np.round(b, 4), "C": np.round(c, 4), "D": np.round(d, 4),
        })
        st.write("#### معادلة كل قطعة:  Conc = A + B·dx + C·dx² + D·dx³  (dx = Abs − من Abs)")
        st.dataframe(seg, use_container_width=True, hide_index=True)
        xs = np.linspace(min(x), max(x), 300)
        show(curve_plot(x, y, xs, [calib.predict_spline(sp, v)[0] for v in xs], "Natural Cubic Spline"))
    with t2:
        simulator(lambda v: calib.predict_spline(sp, v)[0], x)

# ══════════════════════ نتائج Cubic Par A–D ══════════════════════
if res is not None and res["method"] == M_CUBIC:
    x, y, coeffs = res["x"], res["y"], res["coeffs"]
    par_a, par_b, par_c, par_d = coeffs
    y_pred = np.polyval(coeffs, x)
    r2 = calib.compute_r2(y, y_pred)
    res_pct = calib.residual_pct(y, y_pred)
    max_res = calib.max_residual_pct(y, y_pred)
    r2_lin = calib.linearity_r2(x, y)
    grade, grade_color, grade_desc = calib.validation_score(r2, max_res, r2_lin)

    if res["zero"]:
        st.caption("ℹ️ نقطة (0,0) اتشالت من تركيب المنحنى (زي الجهاز).")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 النتائج", "🏆 Validation Score", "⚠️ Outlier Detection",
        "🔄 Delta Check", "🗂️ Archive", "🔍 Sample Simulator"])

    with tab1:
        st.write("### المعاملات (تتكتب في الجهاز):")
        st.latex(r"\text{Conc} = A \cdot \text{Abs}^3 + B \cdot \text{Abs}^2 + C \cdot \text{Abs} + D")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Par A", f"{par_a:.4f}")
        c2.metric("Par B", f"{par_b:.4f}")
        c3.metric("Par C", f"{par_c:.4f}")
        c4.metric("Par D", f"{par_d:.4f}")
        st.caption(f"بدقة 6 أرقام: A={par_a:.6f} · B={par_b:.6f} · C={par_c:.6f} · D={par_d:.6f} · R²={r2:.6f}")
        if len(x) == 4:
            st.success("✅ 4 نقاط فعّالة — المعاملات دي بتطابق اللي بيطلّعه جهاز BK-280.")

        xs = np.linspace(min(x), max(x), 300)

        def _res_lines(ax):
            for i in range(len(x)):
                ax.plot([x[i], x[i]], [y[i], y_pred[i]], "gray", linestyle=":", linewidth=1)
        show(curve_plot(x, y, xs, np.polyval(coeffs, xs), "Calibration Curve (Cubic)", _res_lines))

        st.write("#### جدول المتبقيات (Residuals):")
        st.dataframe(pd.DataFrame({
            "Standard #": [f"Std {i + 1}" for i in range(len(x))],
            "Absorbance": np.round(x, 4), "Actual Conc": np.round(y, 3),
            "Fitted Conc": np.round(y_pred, 3), "Residual": np.round(y - y_pred, 4),
            "Residual %": [("—" if np.isnan(r) else f"{r:.2f}") for r in res_pct],
        }), use_container_width=True, hide_index=True)

        st.write("---")
        st.write("### 📊 One Point End Analysis")
        opf = calib.one_point_factor(x, y)
        if opf is None:
            st.info("محتاج نقطتين موجبتين على الأقل.")
        else:
            x_nz, y_nz, factors, mean_k, cv = opf
            st.dataframe(pd.DataFrame({"Absorbance": np.round(x_nz, 4), "Concentration": np.round(y_nz, 2),
                                       "Factor K (Conc/Abs)": np.round(factors, 2)}),
                         use_container_width=True, hide_index=True)
            f1, f2, f3 = st.columns(3)
            f1.metric("Mean K", f"{mean_k:.2f}")
            f2.metric("CV%", f"{cv:.2f}%")
            if cv <= calib.ONE_POINT_CV:
                f3.metric("الحكم", "✅ ينفع", delta="One Point OK")
                st.success(f"✅ يمكن استخدام One Point End — `Concentration = Absorbance × {mean_k:.2f}`")
            else:
                f3.metric("الحكم", "❌ لا ينفع", delta=f"CV = {cv:.2f}%")
                st.error(f"❌ الفاكتور غير ثابت (CV = {cv:.2f}%) — لازم Cubic Fit.")

    with tab2:
        st.write("### 🏆 Calibration Validation Score")
        st.markdown(f"""<div style='text-align:center; padding:20px; border-radius:12px;
                        background-color:{grade_color}22; border:2px solid {grade_color}'>
                        <h1 style='color:{grade_color}; margin:0'>{grade}</h1>
                        <p style='font-size:18px'>{grade_desc}</p></div>""", unsafe_allow_html=True)
        if len(x) == 4:
            st.caption("ملاحظة: مع 4 نقاط بالظبط المنحنى بيعدّي عليهم كلهم، فـ R² = 1 والانحراف = 0 دايمًا. "
                       "الحكم الحقيقي هنا من الـ Linearity والـ Delta check والـ Controls.")
        v1, v2, v3 = st.columns(3)
        v1.metric("R² (Cubic Fit)", f"{r2:.6f}")
        v2.metric("Max Residual %", f"{max_res:.2f}%")
        v3.metric("Linearity (R² Linear)", f"{r2_lin:.6f}")
        st.dataframe(pd.DataFrame({
            "المعيار": ["R² (Cubic)", "Max Residual %", "Linearity R²"],
            "القيمة الحالية": [f"{r2:.6f}", f"{max_res:.2f}%", f"{r2_lin:.6f}"],
            "Excellent": ["≥ 0.9999", "≤ 1%", "≥ 0.999"], "Good": ["≥ 0.999", "≤ 3%", "≥ 0.995"],
            "Acceptable": ["≥ 0.99", "≤ 5%", "≥ 0.99"], "Failed": ["< 0.99", "> 5%", "< 0.99"],
        }), use_container_width=True, hide_index=True)
        if st.button("💾 حفظ هذه المعايرة في الأرشيف", key="save_archive"):
            entry = {
                "name": cal_name or f"Calibration {datetime.now():%Y-%m-%d %H:%M}",
                "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                "par_a": round(float(par_a), 6), "par_b": round(float(par_b), 6),
                "par_c": round(float(par_c), 6), "par_d": round(float(par_d), 6),
                "r2": round(r2, 6), "max_residual_pct": round(max_res, 2),
                "r2_lin": round(r2_lin, 6), "grade": grade, "n_points": int(len(x)),
            }
            st.session_state.archive.append(entry)
            st.success(f"✅ تم حفظ '{entry['name']}' — نزّل ملف الأرشيف من تبويب Archive عشان ما يضيعش.")

    with tab3:
        st.write("### ⚠️ Auto Outlier Detection")
        thr = st.slider("حد الشذوذ (%)", min_value=5, max_value=30, value=15, step=1)
        outliers = calib.detect_outliers(x, y, coeffs, thr)
        if not outliers:
            st.success("✅ لا توجد نقاط شاذة — كل الـ Standards ضمن الحدود.")
        else:
            st.warning(f"⚠️ تم اكتشاف {len(outliers)} نقطة شاذة!")
            for idx, pct in outliers:
                st.error(f"🔴 **Standard #{idx + 1}** — Abs: {x[idx]:.4f} | Conc: {y[idx]:.3f} | "
                         f"Deviation: **{pct:.1f}%** ← Re-run")
        out_idx = [o[0] for o in outliers]
        xs = np.linspace(min(x), max(x), 300)

        def _band(ax):
            ys = np.polyval(coeffs, xs)
            ax.fill_between(xs, ys * (1 - thr / 100), ys * (1 + thr / 100), alpha=0.1, color="green",
                            label=f"±{thr}% tolerance")
            for i in out_idx:
                ax.scatter(x[i], y[i], color="red", marker="X", s=150, zorder=6)
                ax.annotate(f"Std#{i + 1}", (x[i], y[i]), textcoords="offset points", xytext=(8, 8),
                            color="red", fontsize=9)
        show(curve_plot(x, y, xs, np.polyval(coeffs, xs), "Outlier Detection", _band))

    with tab4:
        st.write("### 🔄 Delta Check — مقارنة المعايرة الجديدة بالقديمة")
        prev = st.session_state.prev_coeffs
        if prev is None:
            st.info("ℹ️ مفيش معايرة سابقة. احسب تاني بعد تعديل البيانات، أو دخّل المعاملات القديمة:")
            m1, m2, m3, m4 = st.columns(4)
            ma = m1.number_input("Par A (قديم)", format="%.6f", key="man_a")
            mb = m2.number_input("Par B (قديم)", format="%.6f", key="man_b")
            mc = m3.number_input("Par C (قديم)", format="%.6f", key="man_c")
            md = m4.number_input("Par D (قديم)", format="%.6f", key="man_d")
            if st.button("تطبيق المعاملات القديمة", key="apply_prev"):
                st.session_state.prev_coeffs = np.array([ma, mb, mc, md])
                st.rerun()
        else:
            deltas = [calib.delta_pct(o, n) for o, n in zip(prev, coeffs)]
            cols = st.columns(4)
            for col, nm, cur, dlt in zip(cols, "ABCD", coeffs, deltas):
                col.metric(f"Par {nm}", f"{cur:.6f}", delta=f"{dlt:+.1f}%",
                           delta_color="inverse" if abs(dlt) > calib.DRIFT_THRESHOLD else "normal")
            max_drift = max(abs(d) for d in deltas)
            if max_drift > calib.DRIFT_THRESHOLD:
                st.error(f"🚨 **Calibration Drift!** أقصى تغيير **{max_drift:.1f}%** "
                         f"(الحد {calib.DRIFT_THRESHOLD:.0f}%)\n\n- راجع الـ Reagent Lot\n"
                         "- راجع صلاحية الـ Standards\n- راجع درجة حرارة التحليل")
            else:
                st.success(f"✅ لا يوجد Drift معتد به — أقصى تغيير {max_drift:.1f}%")
            xs = np.linspace(min(x), max(x), 300)

            def _prev(ax):
                ax.plot(xs, np.polyval(prev, xs), "b--", linewidth=2, label="Previous")
            show(curve_plot(x, y, xs, np.polyval(coeffs, xs), "Delta Check — Curve Comparison", _prev))
            if st.button("🔄 مسح المعايرة القديمة", key="clear_prev"):
                st.session_state.prev_coeffs = None
                st.rerun()

    with tab5:
        st.write("### 🗂️ Archive")
        st.caption("الأرشيف بيتمسح لما الصفحة تتقفل — نزّله كملف JSON وارفعه تاني المرة الجاية.")
        up = st.file_uploader("📤 رفع ملف أرشيف (JSON)", type=["json"], key="arch_up")
        if up is not None and st.button("دمج الملف مع الأرشيف", key="arch_merge"):
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                have = {(a["name"], a["timestamp"]) for a in st.session_state.archive}
                new = [a for a in data if (a.get("name"), a.get("timestamp")) not in have
                       and all(k in a for k in ("par_a", "par_b", "par_c", "par_d", "r2"))]
                st.session_state.archive.extend(new)
                st.success(f"✅ اتضاف {len(new)} معايرة.")
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                st.error(f"الملف مش أرشيف صالح: {e}")
        arch = st.session_state.archive
        if not arch:
            st.info("لا يوجد أرشيف بعد. احفظ المعايرات من تبويب Validation Score.")
        else:
            adf = pd.DataFrame(arch)
            st.dataframe(adf, use_container_width=True, hide_index=True)
            names = [str(a["name"])[:12] for a in arch]
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            for i, (p, colr) in enumerate(zip(["par_a", "par_b", "par_c", "par_d"],
                                              ["blue", "red", "green", "purple"])):
                ax = axes[i // 2][i % 2]
                ax.plot(range(len(arch)), [a[p] for a in arch], "o-", color=colr, linewidth=2)
                ax.set_xticks(range(len(arch)))
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
                ax.set_title(f"Par {p[-1].upper()} over time")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            show(fig)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ الأرشيف JSON (للرفع تاني)", json.dumps(arch, ensure_ascii=False, indent=2),
                               "calibration_archive.json", "application/json")
            d2.download_button("⬇️ الأرشيف CSV (Excel)", adf.to_csv(index=False).encode("utf-8"),
                               "calibration_archive.csv", "text/csv")
            if st.button("🗑️ مسح الأرشيف كله", key="arch_clear"):
                st.session_state.archive = []
                st.rerun()

    with tab6:
        simulator(lambda v: float(np.polyval(coeffs, v)), x)

# ══════════════════════ الفوتر ══════════════════════
st.markdown("---")
st.markdown("<div style='text-align:center; padding:10px 0 20px 0;'>"
            "<p style='margin:4px 0; font-weight:bold;'>🧪 Orange Calibration Tool</p>"
            "<p style='margin:4px 0;'>Developed by Dr / Hussein Ali 2026 For "
            "<span style='color:#E07B00; font-weight:bold;'>Orange Lab</span> 🍊</p></div>",
            unsafe_allow_html=True)
