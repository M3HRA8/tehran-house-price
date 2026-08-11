# 🏠 Tehran House Price Predictor

سامانه هوشمند پیش‌بینی قیمت مسکن تهران با **Random Forest Regression** و رابط **Streamlit**.

## ساختار پروژه

- `app.py` — برنامه اصلی Streamlit
- `data_chunks/` — داده‌های پروژه به‌صورت فشرده و چندبخشی
- `requirements.txt` — وابستگی‌های Python
- `.streamlit/config.toml` — تنظیمات ظاهری Streamlit

برنامه هنگام اولین اجرا داده‌ها را از فایل‌های `data_chunks` بازسازی و پاک‌سازی می‌کند، مدل Random Forest را آموزش می‌دهد و مدل آموزش‌دیده توسط Streamlit Cache نگه داشته می‌شود. بنابراین اجرای سایت به Google Colab وابسته نیست.

## اجرای محلی

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy روی Streamlit Community Cloud

- Repository: `M3HRA8/tehran-house-price`
- Branch: `main`
- Main file path: `app.py`

> قیمت نمایش‌داده‌شده تخمینی است و ممکن است با قیمت واقعی بازار متفاوت باشد.
