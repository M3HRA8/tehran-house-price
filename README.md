# 🏠 Tehran House Price Predictor

سامانه هوشمند پیش‌بینی قیمت مسکن تهران با **Random Forest Regression** و رابط Streamlit.

## فایل‌ها

- `app.py` — برنامه اصلی Streamlit
- `1632300362534233.csv` — دیتاست پروژه
- `requirements.txt` — وابستگی‌های Python
- `.streamlit/config.toml` — تنظیمات ظاهری Streamlit

مدل هنگام شروع برنامه از روی دیتاست آموزش داده می‌شود و نتیجه در حافظه Cache می‌شود؛ بنابراین این نسخه به Google Colab یا فایل مدل جداگانه وابسته نیست.

## اجرای محلی

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy روی Streamlit Community Cloud

Main file path را روی `app.py` قرار دهید.

> قیمت نمایش‌داده‌شده تخمینی است و ممکن است با قیمت واقعی بازار متفاوت باشد.
