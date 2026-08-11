from pathlib import Path

import base64
import gzip
import io

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1) تنظیمات صفحه
# ============================================================

st.set_page_config(
    page_title="سامانه هوشمند پیش‌بینی قیمت مسکن تهران",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2) بارگذاری داده و آموزش مدل
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_CHUNKS_DIR = BASE_DIR / "data_chunks"


@st.cache_resource
def train_model():
    chunk_files = sorted(DATA_CHUNKS_DIR.glob("chunk_*.txt"))
    if not chunk_files:
        raise FileNotFoundError("فایل‌های داده پیدا نشدند.")
    encoded = "".join(
        path.read_text(encoding="ascii").strip()
        for path in chunk_files
    )
    compressed = base64.b64decode(encoded)
    csv_bytes = gzip.decompress(compressed)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # پاک‌سازی مشابه Notebook اصلی پروژه
    df["Area"] = (
        df["Area"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df["Area"] = pd.to_numeric(
        df["Area"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Area", "Address"]
    ).copy()

    df = df[
        (df["Area"] > 0)
        & (df["Area"] <= 1000)
    ].copy()

    for column in [
        "Parking",
        "Warehouse",
        "Elevator",
    ]:
        df[column] = df[column].astype(int)

    df.reset_index(
        drop=True,
        inplace=True,
    )

    addresses = sorted(
        df["Address"]
        .dropna()
        .astype(str)
        .unique()
    )

    df_model = df.drop(
        columns=["Price(USD)"]
    ).copy()

    df_model = pd.get_dummies(
        df_model,
        columns=["Address"],
        drop_first=True,
        dtype=int,
    )

    X = df_model.drop(
        columns=["Price"]
    )

    y = df_model["Price"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return (
        model,
        X.columns.tolist(),
        addresses,
        len(df),
    )


(
    rf_model,
    model_columns,
    addresses,
    DATASET_SIZE,
) = train_model()


# ============================================================
# 3) CSS رابط کاربری
# ============================================================

st.markdown(
    """
    <style>

    /* کل صفحه */
    .stApp {
        direction: rtl;
        background:
            linear-gradient(
                135deg,
                #f8fafc 0%,
                #eef2ff 55%,
                #f0fdf4 100%
            );
        font-family: Tahoma, Arial, sans-serif;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* هدر */
    .hero {
        background:
            linear-gradient(
                135deg,
                #2563eb,
                #4f46e5
            );
        border-radius: 22px;
        padding: 30px 20px;
        margin-bottom: 24px;
        box-shadow:
            0 12px 35px rgba(37, 99, 235, 0.20);
        text-align: center;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 30px;
        font-weight: 900;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #eef2ff;
        font-size: 15px;
        font-weight: 500;
        line-height: 2;
        margin: 0;
    }

    /* کارت فرم */
    .form-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow:
            0 10px 35px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }

    .section-title {
        color: #0f172a;
        font-size: 18px;
        font-weight: 900;
        margin: 2px 0 14px 0;
    }

    /* عنوان‌های ورودی */
    label,
    .stCheckbox label,
    .stSelectbox label,
    .stNumberInput label {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* فیلدهای عددی */
    div[data-baseweb="input"] > div {
        background: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* انتخاب محله */
    div[data-baseweb="select"] > div {
        background: #374151 !important;
        color: #ffffff !important;
        border: 2px solid #1f2937 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* دکمه */
    div.stButton > button {
        width: 100%;
        min-height: 52px;
        color: #ffffff !important;
        font-size: 17px;
        font-weight: 900;
        border-radius: 12px;
        border: none;
        background:
            linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            );
        box-shadow:
            0 7px 18px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 10px 25px rgba(79, 70, 229, 0.35);
        color: #ffffff !important;
        border: none;
    }

    /* نتیجه */
    .result-box {
        background:
            linear-gradient(
                135deg,
                #f0f7ff,
                #eef2ff
            );
        color: #0f172a;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 20px;
        margin-top: 18px;
        box-shadow:
            0 5px 18px rgba(30, 64, 175, 0.08);
        text-align: center;
    }

    .result-title {
        font-size: 15px;
        font-weight: 600;
        color: #475569;
        margin-bottom: 8px;
    }

    .result-price {
        font-size: 34px;
        font-weight: 900;
        color: #1d4ed8;
        margin: 5px 0 7px 0;
    }

    .result-price span {
        font-size: 16px;
        color: #475569;
        font-weight: 600;
    }

    .billion-badge {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .result-grid {
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(140px, 1fr));
        gap: 10px;
        margin-top: 5px;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 8px;
    }

    .result-card-label {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .result-card-value {
        color: #0f172a;
        font-size: 16px;
        font-weight: 800;
    }

    .features-box {
        margin-top: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 11px;
        border-radius: 10px;
        color: #334155;
        font-size: 14px;
        font-weight: 600;
    }

    .note {
        margin-top: 15px;
        font-size: 12px;
        color: #64748b;
        line-height: 1.9;
    }

    /* خطا */
    .error-box {
        text-align: center;
        padding: 20px;
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 14px;
        color: #9a3412;
        font-size: 15px;
        line-height: 2;
        margin-top: 16px;
    }

    .error-icon {
        font-size: 30px;
        margin-bottom: 7px;
    }

    .error-text {
        font-weight: 700;
        color: #9a3412;
    }

    /* درباره سامانه */
    .model-info {
        background: #ffffff;
        color: #334155;
        padding: 20px 22px;
        margin-top: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow:
            0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .model-title {
        color: #0f172a;
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .info-text {
        color: #475569;
        line-height: 2.2;
    }

    .info-text b {
        color: #1e293b;
    }

    /* فوتر */
    .footer {
        width: 100%;
        text-align: center;
        direction: rtl;
        margin-top: 18px;
        padding: 18px 10px;
        color: #475569;
        font-size: 14px;
    }

    .footer-highlight {
        color: #2563eb;
        font-weight: 900;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-left: 10px;
            padding-right: 10px;
        }

        .hero {
            padding: 22px 13px;
            border-radius: 17px;
        }

        .hero h1 {
            font-size: 21px;
            line-height: 1.8;
        }

        .hero p {
            font-size: 13px;
        }

        .form-card {
            padding: 15px;
        }

        .result-box {
            padding: 12px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4) توابع کمکی
# ============================================================

def error_html(message, icon="⚠️"):
    return f"""
    <div class="error-box">
        <div class="error-icon">{icon}</div>
        <div class="error-text">{message}</div>
    </div>
    """


def predict_house_price(
    area,
    room,
    parking,
    warehouse,
    elevator,
    address,
):

    if area is None:
        return None, "لطفاً متراژ خانه را وارد کنید."

    if room is None:
        return None, "لطفاً تعداد اتاق را وارد کنید."

    if address is None or str(address).strip() == "":
        return None, "لطفاً محله را انتخاب کنید."

    try:
        area = float(area)
        room = float(room)
    except (TypeError, ValueError):
        return None, "مقادیر واردشده معتبر نیستند."

    if not np.isfinite(area):
        return None, "متراژ واردشده معتبر نیست."

    if not np.isfinite(room):
        return None, "تعداد اتاق واردشده معتبر نیست."

    if area < 30:
        return None, "متراژ خانه نمی‌تواند کمتر از ۳۰ متر باشد."

    if area > 1000:
        return None, "حداکثر متراژ قابل قبول ۱۰۰۰ متر است."

    if room < 0:
        return None, "تعداد اتاق نمی‌تواند منفی باشد."

    if room > 10:
        return None, "حداکثر تعداد اتاق قابل قبول ۱۰ است."

    if room != int(room):
        return None, "تعداد اتاق باید عدد صحیح باشد."

    area = int(area)
    room = int(room)
    address = str(address)

    if address not in addresses:
        return None, "محله انتخاب‌شده در اطلاعات پروژه وجود ندارد."

    new_house = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns,
    )

    if "Area" in new_house.columns:
        new_house["Area"] = area

    if "Room" in new_house.columns:
        new_house["Room"] = room

    if "Parking" in new_house.columns:
        new_house["Parking"] = 1 if parking else 0

    if "Warehouse" in new_house.columns:
        new_house["Warehouse"] = 1 if warehouse else 0

    if "Elevator" in new_house.columns:
        new_house["Elevator"] = 1 if elevator else 0

    address_column = "Address_" + address

    if address_column in new_house.columns:
        new_house[address_column] = 1

    try:
        predicted_price = rf_model.predict(new_house)[0]
    except Exception as exc:
        print("Prediction Error:", exc)
        return None, (
            "خطایی در انجام پیش‌بینی رخ داد. "
            "لطفاً اطلاعات واردشده را بررسی کنید."
        )

    if not np.isfinite(predicted_price) or predicted_price < 0:
        return None, (
            "امکان ارائه پیش‌بینی معتبر برای این مشخصات وجود ندارد."
        )

    price_per_meter = predicted_price / area

    features = []

    if parking:
        features.append("🚗 پارکینگ")

    if warehouse:
        features.append("📦 انباری")

    if elevator:
        features.append("⚙ آسانسور")

    if len(features) == 0:
        features_text = "هیچ‌کدام از امکانات انتخاب نشده است"
    else:
        features_text = " &nbsp;&nbsp; | &nbsp;&nbsp; ".join(features)

    result_html = f"""
    <div class="result-box">

        <div class="result-title">
            💰 قیمت تقریبی ملک
        </div>

        <div class="result-price">
            {predicted_price:,.0f}
            <span>تومان</span>
        </div>

        <div class="billion-badge">
            حدود
            {predicted_price / 1_000_000_000:.2f}
            میلیارد تومان
        </div>

        <div class="result-grid">

            <div class="result-card">
                <div class="result-card-label">
                    📐 متراژ
                </div>
                <div class="result-card-value">
                    {area:,} متر
                </div>
            </div>

            <div class="result-card">
                <div class="result-card-label">
                    🛏 تعداد اتاق
                </div>
                <div class="result-card-value">
                    {room} اتاق
                </div>
            </div>

            <div class="result-card">
                <div class="result-card-label">
                    💵 قیمت هر متر
                </div>
                <div class="result-card-value">
                    {price_per_meter:,.0f} تومان
                </div>
            </div>

            <div class="result-card">
                <div class="result-card-label">
                    📍 محله
                </div>
                <div class="result-card-value">
                    {address}
                </div>
            </div>

        </div>

        <div class="features-box">
            {features_text}
        </div>

        <div class="note">
            این قیمت توسط مدل یادگیری ماشین تخمین زده شده
            و ممکن است با قیمت واقعی بازار متفاوت باشد.
        </div>

    </div>
    """

    return result_html, None


# ============================================================
# 5) هدر
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>
            🏠 سامانه هوشمند پیش‌بینی قیمت مسکن
        </h1>

        <p>
            مشخصات ملک موردنظر خود را وارد کنید
            <br>
            تا هوش مصنوعی قیمت تقریبی آن را
            براساس داده‌های مسکن تهران
            پیش‌بینی کند
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6) فرم
# ============================================================

st.markdown(
    '<div class="form-card">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">🏡 مشخصات ملک</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "📐 متراژ خانه",
        min_value=30,
        max_value=1000,
        value=100,
        step=1,
    )

with col2:
    room = st.number_input(
        "🛏 تعداد اتاق",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
    )

address = st.selectbox(
    "📍 محله ملک",
    options=addresses,
    index=(
        addresses.index("Punak")
        if "Punak" in addresses
        else 0
    ),
    help="محله موردنظر را از فهرست انتخاب کنید",
)

st.markdown(
    '<div class="section-title" style="margin-top:15px;">✨ امکانات ملک</div>',
    unsafe_allow_html=True,
)

f1, f2, f3 = st.columns(3)

with f1:
    parking = st.checkbox(
        "🚗 پارکینگ",
        value=True,
    )

with f2:
    warehouse = st.checkbox(
        "📦 انباری",
        value=True,
    )

with f3:
    elevator = st.checkbox(
        "⚙ آسانسور",
        value=True,
    )

clicked = st.button(
    "🔍 محاسبه قیمت تقریبی ملک",
    use_container_width=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# 7) نتیجه
# ============================================================

if clicked:

    result, error = predict_house_price(
        area=area,
        room=room,
        parking=parking,
        warehouse=warehouse,
        elevator=elevator,
        address=address,
    )

    if error:
        st.markdown(
            error_html(error),
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            result,
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        """
        <div class="result-box">

            <div style="
                font-size:38px;
                margin-bottom:8px;
            ">
                💰
            </div>

            <div style="
                color:#334155;
                font-size:15px;
                font-weight:700;
                line-height:2;
            ">
                اطلاعات ملک را وارد کرده
                <br>
                و روی دکمه
                <span style="
                    color:#2563eb;
                    font-weight:900;
                ">
                    محاسبه قیمت
                </span>
                کلیک کنید.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8) درباره سامانه
# ============================================================

st.markdown(
    f"""
    <div class="model-info">

        <div class="model-title">
            🤖 درباره سامانه
        </div>

        <div class="info-text">

            <b>مدل مورد استفاده:</b>
            <span dir="ltr">
                Random Forest Regression
            </span>

            <br>

            <b>تعداد داده‌های مورد استفاده:</b>
            <span>
                {DATASET_SIZE:,} ملک
            </span>

            <br>

            <b>نوع مسئله:</b>
            <span>
                پیش‌بینی قیمت مسکن
            </span>

            <br>

            <b>محدوده:</b>
            <span>
                شهر تهران
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9) فوتر
# ============================================================

st.markdown(
    """
    <div class="footer">

        طراحی و توسعه با

        <span class="footer-highlight">
            Python
        </span>

        🐍

        و

        <span class="footer-highlight">
            Machine Learning
        </span>

        🤖

    </div>
    """,
    unsafe_allow_html=True,
)
