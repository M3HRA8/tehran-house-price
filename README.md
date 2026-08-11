# 🏠 Tehran House Price Predictor

سامانه هوشمند پیش‌بینی قیمت مسکن تهران با استفاده از **Random Forest Regression** و رابط کاربری **Streamlit**.

🌐 **نسخه آنلاین:** https://tehran-house-price.streamlit.app

## ✨ امکانات

- پیش‌بینی قیمت تقریبی ملک بر اساس مشخصات واردشده
- انتخاب محله از میان محله‌های موجود در دیتاست
- درنظرگرفتن متراژ، تعداد اتاق، پارکینگ، انباری و آسانسور
- نمایش قیمت کل و قیمت تقریبی هر متر
- رابط فارسی و راست‌به‌چپ
- طراحی Responsive برای دسکتاپ و موبایل
- اعتبارسنجی ورودی‌ها و نمایش پیام خطای مناسب
- اجرای مستقل از Google Colab روی Streamlit Community Cloud

## 🤖 مدل یادگیری ماشین

در روند توسعه پروژه دو مدل بررسی شدند:

- Linear Regression
- Random Forest Regression

Random Forest عملکرد بهتری داشت و به‌عنوان مدل نهایی انتخاب شد.

نتایج آزمایش پروژه تقریباً به این صورت بود:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 2.00 میلیارد تومان | 4.26 میلیارد تومان | 0.70 |
| Random Forest | 1.35 میلیارد تومان | 3.44 میلیارد تومان | 0.80 |

> R² برابر 0.80 به معنی «دقت 80 درصد» نیست؛ این معیار میزان توضیح تغییرات متغیر هدف توسط مدل را نشان می‌دهد.

## 📁 ساختار Repository

```text
tehran-house-price/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data_chunks/
    ├── chunk_01.txt
    ├── chunk_02.txt
    ├── chunk_03.txt
    └── chunk_04.txt
```

- `app.py` — برنامه اصلی Streamlit و منطق پیش‌بینی
- `data_chunks/` — دیتاست پروژه به‌صورت فشرده و چندبخشی
- `requirements.txt` — وابستگی‌های Python
- `.streamlit/config.toml` — تنظیمات ظاهری Streamlit

برنامه هنگام اولین اجرا داده‌ها را بازسازی و پاک‌سازی می‌کند، سپس مدل Random Forest را آموزش می‌دهد. مدل آموزش‌دیده توسط Streamlit Cache نگه داشته می‌شود تا در اجرای مجدد از آموزش غیرضروری جلوگیری شود.

## 🧹 پیش‌پردازش داده‌ها

مراحل اصلی آماده‌سازی داده شامل موارد زیر است:

1. تبدیل `Area` به مقدار عددی
2. حذف رکوردهای فاقد `Address`
3. حذف متراژهای غیرعادی بیشتر از 1000 متر
4. تبدیل `Parking`، `Warehouse` و `Elevator` به 0 و 1
5. حذف `Price(USD)` از ورودی مدل برای جلوگیری از Data Leakage
6. تبدیل `Address` با One-Hot Encoding

پس از پاک‌سازی، **3451 ملک** در پروژه مورد استفاده قرار گرفت.

## 💻 اجرای محلی

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy

این پروژه روی Streamlit Community Cloud اجرا می‌شود:

- Repository: `M3HRA8/tehran-house-price`
- Branch: `main`
- Main file path: `app.py`

## ⚠️ محدودیت

قیمت مسکن علاوه بر ویژگی‌های موجود در دیتاست به عوامل دیگری مانند سن ساختمان، طبقه، کیفیت ساخت، سال ساخت، موقعیت دقیق و شرایط روز بازار وابسته است. بنابراین خروجی این سامانه **تخمینی** است و نباید به‌عنوان قیمت قطعی معامله در نظر گرفته شود.

---

**Technologies:** Python · Pandas · NumPy · Scikit-learn · Random Forest · Streamlit
