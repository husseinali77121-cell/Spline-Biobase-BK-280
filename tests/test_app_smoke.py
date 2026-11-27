"""Smoke test: بيشغّل streamlit_app.py من أوله لآخره بـ Streamlit وهمي،
في الطريقتين ومع ضغط كل الأزرار، عشان يمسك أي خطأ في الواجهة."""
import os
import runpy
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "streamlit_app.py")


class _Stop(Exception):
    pass


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):          # col.metric(...) / col.button(...)
        return getattr(_FAKE, name)


class _State(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v


class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _State()
        self.secrets = {}
        self.cfg = {}
        self.errors = []

    def _v(self, label, key, default):
        return self.cfg.get(key, self.cfg.get(label, default))

    def set_page_config(self, **k): pass
    def markdown(self, *a, **k): pass
    write = title = info = success = warning = caption = latex = subheader = markdown
    def error(self, msg, *a, **k): self.errors.append(msg)
    def metric(self, *a, **k): pass
    def dataframe(self, *a, **k): pass
    def pyplot(self, fig, *a, **k): pass
    def download_button(self, *a, **k): return False
    def file_uploader(self, label, key=None, **k): return self._v(label, key, None)
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]
    def tabs(self, names): return [_Ctx() for _ in names]
    def stop(self): raise _Stop()
    def rerun(self): raise _Stop()
    def radio(self, label, options, key=None, **k): return self._v(label, key, options[0])
    def button(self, label, key=None, **k): return bool(self._v(label, key, False))
    def text_input(self, label, key=None, **k): return self._v(label, key, "")
    def text_area(self, label, key=None, **k): return self._v(label, key, "")
    def number_input(self, label, key=None, **k): return self._v(label, key, 0.0)
    def slider(self, label, value=None, key=None, **k): return self._v(label, key, value)
    def data_editor(self, df, key=None, **k): return self._v("editor", key, df)


_FAKE = FakeSt()


def run(cfg):
    _FAKE.cfg = cfg
    _FAKE.errors = []
    sys.modules["streamlit"] = _FAKE
    sys.path.insert(0, ROOT)
    try:
        runpy.run_path(APP, run_name="__main__")
    except _Stop:
        pass
    return _FAKE


@pytest.fixture(autouse=True)
def fresh():
    _FAKE.session_state.clear()
    _FAKE.secrets = {}
    yield


CALC = "🔬 احسب (Calculate)"


def test_cubic_full_flow():
    st = run({CALC: True})
    r = st.session_state["result"]
    assert r["method"].startswith("Cubic") and len(r["coeffs"]) == 4
    # تاني حساب: Delta check + أرشيف + محاكي جوه/برا المدى + Batch
    st = run({CALC: True, "save_archive": True, "test_absorbance": 0.05, "batch_calc": True,
              "قيم Absorbance (كل قيمة في سطر):": "0.02\n0.3\n0.1"})
    assert st.session_state["prev_coeffs"] is not None
    assert len(st.session_state["archive"]) == 1
    st = run({"test_absorbance": 0.5})                      # Extrapolation
    # الرسالة الوحيدة المتوقعة: One-point CV عالي (بيانات Vit-D مش خطية) — مش خطأ في البرنامج
    assert all("الفاكتور غير ثابت" in e for e in st.errors)


def test_spline_flow():
    st = run({"method": "Natural spline — قطع بين النقاط (Vit-D)", CALC: True,
              "test_absorbance": 0.06, "batch_calc": True,
              "قيم Absorbance (كل قيمة في سطر):": "0.01\n0.2"})
    r = st.session_state["result"]
    assert r["method"].startswith("Natural") and len(r["x"]) == 5
    assert not st.errors


def test_switch_method_asks_to_recalculate():
    run({CALC: True})
    st = run({"method": "Natural spline — قطع بين النقاط (Vit-D)"})
    assert st.session_state["result"]["method"].startswith("Cubic")   # ما اتحسبش لوحده
    assert not st.errors


def test_bad_input_shows_error_not_crash():
    import pandas as pd
    bad = pd.DataFrame({"Absorbance": [0.0, 0.1, 0.1], "Concentration": [0.0, 5, "x"]})
    st = run({CALC: True, "editor": bad})
    assert st.errors and st.session_state["result"] is None


def test_manual_previous_and_clear():
    run({CALC: True})
    st = run({"man_a": 1.0, "man_b": 2.0, "man_c": 3.0, "man_d": 4.0, "apply_prev": True})
    np.testing.assert_allclose(st.session_state["prev_coeffs"], [1, 2, 3, 4])
    st = run({"clear_prev": True})
    assert st.session_state["prev_coeffs"] is None


def test_login_only_when_secrets_set():
    _FAKE.secrets = {"allowed_emails": {"emails": ["a@lab.com"]}, "app": {"password": "p"}}
    st = run({CALC: True})
    assert st.session_state["authenticated"] is False and st.session_state.get("result") is None
    st = run({"login_email": "A@lab.com ", "login_pass": "p", "دخول ▶": True})
    assert st.session_state["authenticated"] is True


def test_archive_clear():
    run({CALC: True, "save_archive": True})
    st = run({"arch_clear": True})
    assert st.session_state["archive"] == []


class _Up:
    def __init__(self, b): self.b = b
    def getvalue(self): return self.b


def test_archive_upload_merge_and_bad_file():
    import json
    run({CALC: True, "save_archive": True})
    good = [{"name": "Old run", "timestamp": "2026-01-01 10:00:00", "par_a": 1, "par_b": 2,
             "par_c": 3, "par_d": 4, "r2": 1, "max_residual_pct": 0, "r2_lin": 1,
             "grade": "x", "n_points": 4}]
    st = run({"arch_up": _Up(json.dumps(good).encode()), "arch_merge": True})
    assert len(st.session_state["archive"]) == 2
    st = run({"arch_up": _Up(json.dumps(good).encode()), "arch_merge": True})   # مفيش تكرار
    assert len(st.session_state["archive"]) == 2
    st = run({"arch_up": _Up(b"not json"), "arch_merge": True})
    assert st.errors and len(st.session_state["archive"]) == 2
    st = run({"arch_up": _Up(b'{"a": 1}'), "arch_merge": True})                 # JSON مش قايمة
    assert st.errors
