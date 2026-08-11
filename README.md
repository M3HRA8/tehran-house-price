# 🏠 Tehran House Price Predictor

سامانه هوشمند پیش‌بینی قیمت مسکن تهران با استفاده از **Random Forest Regression** و رابط کاربری فارسی و Responsive.

🌐 **نسخه آنلاین:** https://m3hra8.github.io/tehran-house-price/

## ✨ امکانات

- پیش‌بینی قیمت تقریبی ملک بر اساس متراژ، تعداد اتاق و محله
- درنظرگرفتن پارکینگ، انباری و آسانسور
- نمایش قیمت کل و قیمت تقریبی هر متر
- رابط فارسی و راست‌به‌چپ
- طراحی مناسب دسکتاپ و موبایل
- اعتبارسنجی ورودی‌ها و نمایش پیام خطای مناسب
- اجرای مستقیم مدل در مرورگر، بدون نیاز به Google Colab یا سرور Python در زمان استفاده

## 🤖 مدل یادگیری ماشین

در مرحله ارزیابی دو مدل بررسی شدند:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | حدود 2.00 میلیارد تومان | حدود 4.26 میلیارد تومان | 0.70 |
| Random Forest | حدود 1.35 میلیارد تومان | حدود 3.44 میلیارد تومان | 0.80 |

Random Forest عملکرد بهتری داشت و به‌عنوان مدل نهایی انتخاب شد.

> R² برابر 0.80 به معنی «دقت 80 درصد» نیست؛ این معیار میزان توضیح تغییرات متغیر هدف توسط مدل را نشان می‌دهد.

### مدل نهایی سایت

مدل Production سایت دارای مشخصات زیر است:

- `RandomForestRegressor`
- `n_estimators=300`
- `random_state=42`
- آموزش روی تمام **3451 رکورد پاک‌سازی‌شده**
- تبدیل دقیق ساختار درخت‌ها به فرمت مرورگری با مقادیر `float64`

معیارهای ارزیابی از مدل جداگانه‌ای با تقسیم Train/Test به نسبت 80/20 محاسبه شده‌اند؛ مدل نهایی سایت برای استفاده از تمام داده‌های موجود دوباره روی کل دیتاست پاک‌سازی‌شده آموزش داده می‌شود.

## 🧹 پیش‌پردازش داده‌ها

مراحل اصلی آماده‌سازی داده:

1. تبدیل `Area` به مقدار عددی
2. حذف رکوردهای فاقد `Address`
3. حذف متراژهای غیرعادی بیشتر از 1000 متر
4. تبدیل `Parking`، `Warehouse` و `Elevator` به 0 و 1
5. حذف `Price(USD)` از ورودی مدل برای جلوگیری از Data Leakage
6. تبدیل `Address` با One-Hot Encoding

پس از پاک‌سازی، **3451 ملک** در پروژه مورد استفاده قرار گرفت.

## 📁 ساختار Repository

```text
tehran-house-price/
├── README.md
├── requirements.txt
├── .gitignore
├── data_chunks/
│   ├── chunk_01.txt
│   ├── chunk_02.txt
│   ├── chunk_03.txt
│   └── chunk_04.txt
├── scripts/
│   └── build_static.py
├── static/
│   └── index.html
└── .github/
    └── workflows/
        └── pages.yml
```

- `data_chunks/` — دیتاست پروژه به‌صورت فشرده و چندبخشی
- `scripts/build_static.py` — بازسازی داده، آموزش مدل 300 درختی و ساخت مدل قابل اجرا در مرورگر
- `static/index.html` — رابط کاربری سایت و منطق پیش‌بینی JavaScript
- `.github/workflows/pages.yml` — Build و Deploy خودکار روی GitHub Pages
- `requirements.txt` — کتابخانه‌های لازم برای مرحله Build

## 💻 اجرای محلی

```bash
pip install -r requirements.txt
python scripts/build_static.py
python -m http.server 8000 --directory site
```

سپس در مرورگر آدرس `http://localhost:8000` را باز کنید.

## 🚀 Deployment

هر Push روی شاخه `main` باعث اجرای GitHub Actions می‌شود. Workflow پروژه:

1. داده‌ها را بازسازی و پاک‌سازی می‌کند.
2. مدل Random Forest نهایی را آموزش می‌دهد.
3. مدل را برای اجرای مستقیم در JavaScript تبدیل می‌کند.
4. سایت نهایی را روی GitHub Pages منتشر می‌کند.

## ⚠️ محدودیت

قیمت مسکن علاوه بر ویژگی‌های موجود در دیتاست به عواملی مانند سن ساختمان، طبقه، کیفیت ساخت، سال ساخت، موقعیت دقیق و شرایط روز بازار وابسته است. بنابراین خروجی این سامانه **تخمینی** است و نباید به‌عنوان قیمت قطعی معامله در نظر گرفته شود.

---

**Technologies:** Python · Pandas · NumPy · Scikit-learn · Random Forest · HTML · CSS · JavaScript · GitHub Pages
