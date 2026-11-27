# 🧪 Orange Calibration Tool — BK-280 Spline

برنامج واحد لحساب معاملات المعايرة لجهاز **Biobase BK-280**، مدموج من 3 برامج قديمة:

| البرنامج القديم | اللي اتنقل منه |
|---|---|
| Spline-Biobase-BK-280 | Par A·B·C·D زي شاشة الجهاز (نقطة 0,0 مستبعدة) |
| My-spline-new | Validation score · Outliers · One-point factor · Delta check · Archive · Sample/Batch simulator · الدخول |
| Vit-D_concentraion | Natural cubic spline (اختيار تاني في أول الصفحة) |

النسخ القديمة متأرشفة (read-only) على GitHub ومحدش مسحها.

## طريقتين للحساب
- **Cubic — Par A–D:** `Conc = A·Abs³ + B·Abs² + C·Abs + D`. دي المعاملات اللي بتتكتب في الجهاز.
  مع 4 نقاط (بعد استبعاد الصفر) النتيجة **مطابقة بالظبط** للبرنامج القديم ولشاشة الجهاز.
  مع أكتر من 4 نقاط بيعمل least-squares (البرنامج القديم كان بيقف بخطأ).
- **Natural spline:** منحنى بيعدّي على كل Standard، لكل قطعة معادلتها. نفس حسابات Vit-D بالظبط.

## الدخول (اختياري)
لو عايز تقفل البرنامج بباسورد، حط في Streamlit → Settings → Secrets:
```toml
[allowed_emails]
emails = ["name@lab.com"]

[app]
password = "..."

[contact]          # اختياري — بيظهر في صفحة الدخول
name  = "Dr / Hussein Ali"
phone = "..."
email = "..."
```
من غير الـ Secrets دي البرنامج بيفتح على طول.

## الأرشيف
بيتحفظ طول ما الصفحة مفتوحة. نزّله **JSON** من تبويب Archive وارفعه تاني المرة الجاية (أو CSV للإكسل).

## التشغيل والاختبارات
```
pip install -r requirements.txt
streamlit run streamlit_app.py
python -m pytest -q tests        # 26 اختبار: الحسابات مطابقة للبرامج القديمة + تشغيل الواجهة كاملة
```
