import html
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyowm
from pyowm.commons.exceptions import NotFoundError, UnauthorizedError
import requests
import streamlit as st
from groq import Groq
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Agriculture System",
    layout="wide",
    initial_sidebar_state="expanded",
)

YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# Uploaders / rights holders known to block embedded playback
BLOCKED_VIDEO_TERMS = ("etpl", "etv", "blocked", "restricted")

# Curated fallback videos when API results are restricted or quota is unavailable
CURATED_FARM_VIDEOS = {
    "cotton": [
        {"video_id": "8Q7q5-9j9dQ", "title": "Cotton Crop Cultivation and Nutrient Management Guide"},
        {"video_id": "gG3Q8p7zW2Y", "title": "Balanced NPK Fertilizer and Pest Care for Cotton"},
    ],
    "rice": [
        {"video_id": "Z6S6u4k8g1M", "title": "Paddy Cultivation, Water Depth and Fertilizer Schedule"},
        {"video_id": "2c8Z7k5mL9A", "title": "High Yield Rice Crop Management Practices"},
    ],
    "paddy": [
        {"video_id": "Z6S6u4k8g1M", "title": "Paddy Cultivation, Water Depth and Fertilizer Schedule"},
        {"video_id": "2c8Z7k5mL9A", "title": "High Yield Rice Crop Management Practices"},
    ],
    "maize": [
        {"video_id": "3H9L2x8vN4P", "title": "Maize Farming, Soil Preparation and Fertilizer Tips"},
    ],
    "default": [
        {"video_id": "Z6S6u4k8g1M", "title": "Soil Nutrition and Crop Management Overview"},
        {"video_id": "8Q7q5-9j9dQ", "title": "Field Fertilizer Application and Irrigation Guide"},
    ],
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    /* Sidebar & main radio pills: light pill with dark text for both light and dark modes */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0.4rem;
    }
    div[data-testid="stRadio"] label {
        background-color: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.8rem !important;
        transition: all 0.18s ease;
    }
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label div {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #2D6A4F !important;
        background-color: #DCFCE7 !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.14s ease, box-shadow 0.14s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.14);
    }

    .header-box {
        background: linear-gradient(120deg, #143628 0%, #1B4332 48%, #2D6A4F 100%);
        padding: 2rem 2.25rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.4rem;
    }
    .header-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        color: #E9F5EE;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }
    .header-box h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: #FFFFFF !important;
    }
    .header-box p {
        font-size: 0.98rem;
        color: #E9F5EE;
        margin: 0;
        max-width: 780px;
        line-height: 1.5;
    }

    .page-heading {
        background: #F1F8F5;
        border-left: 4px solid #2D6A4F;
        padding: 0.9rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .page-heading h2 {
        margin: 0;
        color: #1B4332 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .page-heading p {
        margin: 0.2rem 0 0 0;
        color: #475569 !important;
        font-size: 0.9rem;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #2D6A4F;
        border-radius: 10px;
        padding: 1.15rem;
        margin-bottom: 0.55rem;
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .info-card:hover {
        border-color: #40916C;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
    }
    .info-card-title {
        font-size: 1.04rem;
        font-weight: 700;
        color: #1E293B !important;
        margin-bottom: 0.3rem;
    }
    .info-card-text {
        font-size: 0.87rem;
        color: #64748B !important;
        line-height: 1.45;
        margin: 0;
    }

    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: #64748B !important;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1B4332 !important;
        margin: 0;
    }
    .stat-note {
        font-size: 0.78rem;
        color: #40916C !important;
        margin-top: 0.2rem;
    }

    .result-box {
        background: #E8F5E9;
        border: 1px solid #66BB6A;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #1B4332 !important;
        margin-bottom: 1rem;
    }
    .result-box h3 {
        margin: 0.2rem 0 0.4rem 0;
        color: #1B4332 !important;
        font-size: 1.4rem;
    }
    .result-box p {
        margin: 0;
        color: #2D6A4F !important;
        font-size: 0.93rem;
    }

    .weather-box {
        background: linear-gradient(135deg, #1E3A5F 0%, #2B5876 100%);
        color: #FFFFFF !important;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
    }
    .weather-box h2, .weather-box p {
        color: #FFFFFF !important;
    }

    .slide-caption {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        text-align: center;
        color: #334155;
        font-size: 0.9rem;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .video-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.85rem;
    }

    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.84rem;
        padding-top: 1.25rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 2.5rem;
    }

    /* Dark mode card adjustments */
    @media (prefers-color-scheme: dark) {
        .info-card, .stat-card, .video-card {
            background-color: #1E293B !important;
            border-color: #334155 !important;
        }
        .info-card-title {
            color: #F8FAFC !important;
        }
        .info-card-text, .stat-label {
            color: #CBD5E1 !important;
        }
        .stat-value {
            color: #86EFAC !important;
        }
        .page-heading {
            background-color: #132A20 !important;
        }
        .page-heading h2 {
            color: #A7F3D0 !important;
        }
        .page-heading p {
            color: #CBD5E1 !important;
        }
        .slide-caption {
            background-color: #1E293B !important;
            border-color: #334155 !important;
            color: #E2E8F0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_crop_recommendation_data():
    return pd.read_csv("Crop_recommendation.csv")


@st.cache_resource
def load_crop_recommendation_model():
    df = load_crop_recommendation_data()
    if os.path.exists("new_rf_model.pickle"):
        try:
            return joblib.load("new_rf_model.pickle"), df
        except Exception:
            pass
    features = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    target = df["label"]
    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=2
    )
    rf = RandomForestClassifier(n_estimators=25, random_state=42)
    rf.fit(x_train, y_train)
    return rf, df


@st.cache_data
def load_yield_data():
    data = pd.read_csv("crop_data.csv")
    x_cols = pd.get_dummies(data.drop(["CROP_PRICE", "CROP"], axis=1)).columns
    return data, x_cols


@st.cache_resource
def load_yield_model():
    data, _ = load_yield_data()
    if os.path.exists("yield_prediction_model.pkl"):
        try:
            return joblib.load("yield_prediction_model.pkl")
        except Exception:
            pass
    x = pd.get_dummies(data.drop(["CROP_PRICE", "CROP"], axis=1)).astype(float)
    y = data["CROP_PRICE"]
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(x, y)
    return reg


def get_home_slides():
    default_titles = {
        1: "Smart Irrigation and Precision Field Monitoring",
        2: "Soil Nutrient Analysis and Crop Suitability",
        3: "Automated Weather and Irrigation Management",
        4: "Data-Driven Yield and Harvest Planning",
    }

    slides = []
    if os.path.exists("agri.jpeg"):
        slides.append(("agri.jpeg", default_titles[1]))

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    if os.path.isdir("slides"):
        for fname in sorted(os.listdir("slides")):
            if fname.lower().endswith(valid_exts):
                fpath = os.path.join("slides", fname)
                slide_num = len(slides) + 1
                title = default_titles.get(slide_num, f"Smart Agriculture Overview {slide_num}")
                slides.append((fpath, title))

    for i in range(2, 10):
        for ext in valid_exts:
            candidate = f"agri{i}{ext}"
            if os.path.exists(candidate):
                slide_num = len(slides) + 1
                title = default_titles.get(slide_num, f"Smart Agriculture Overview {slide_num}")
                slides.append((candidate, title))

    if len(slides) < 2:
        fallback_samples = [
            ("images/rice.jpg", "Paddy Cultivation and Water Management"),
            ("images/maize.jpg", "Maize Crop Nutrition and Yield Monitoring"),
            ("images/cotton.jpg", "Cotton Soil Suitability and Growth"),
            ("images/coffee.jpg", "Plantation Crop Climate Analysis"),
        ]
        for path, caption in fallback_samples:
            if os.path.exists(path):
                slides.append((path, caption))

    return slides


def _is_embed_blocked(title, channel_title=""):
    combined = f"{title} {channel_title}".lower()
    return any(term in combined for term in BLOCKED_VIDEO_TERMS)


def _is_video_embeddable(video_id):
    if not video_id:
        return False
    oembed_url = (
        f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        res = requests.get(oembed_url, timeout=6)
        return res.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def _query_youtube_api(query, max_results=6):
    if not YOUTUBE_API_KEY:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoEmbeddable": "true",
        "videoSyndicated": "true",
        "safeSearch": "strict",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        videos = []
        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            raw_title = html.unescape(snippet.get("title", "Farming Guide"))
            channel = html.unescape(snippet.get("channelTitle", ""))
            if vid and not _is_embed_blocked(raw_title, channel):
                if _is_video_embeddable(vid):
                    videos.append({"video_id": vid, "title": raw_title})
        return videos
    except requests.RequestException:
        return []


def fetch_youtube_videos(primary_query, fallback_query=None, crop_key="default", max_results=4):
    videos = _query_youtube_api(primary_query, max_results=6)
    if len(videos) < 2 and fallback_query and fallback_query != primary_query:
        extra = _query_youtube_api(fallback_query, max_results=6)
        seen_ids = {v["video_id"] for v in videos}
        for v in extra:
            if v["video_id"] not in seen_ids:
                videos.append(v)
                seen_ids.add(v["video_id"])
            if len(videos) >= max_results:
                break

    if not videos:
        curated = CURATED_FARM_VIDEOS.get(crop_key.lower(), CURATED_FARM_VIDEOS["default"])
        return curated[:max_results]

    return videos[:max_results]


def render_result_videos(title, primary_query, fallback_query=None, crop_key="default", max_results=4):
    st.subheader(title)
    st.caption(f"Auto-matched tutorials for: {primary_query}")
    with st.spinner("Loading verified farming videos..."):
        videos = fetch_youtube_videos(
            primary_query,
            fallback_query=fallback_query,
            crop_key=crop_key,
            max_results=max_results,
        )

    if videos:
        for i in range(0, len(videos), 2):
            v_cols = st.columns(2)
            for j, video in enumerate(videos[i : i + 2]):
                with v_cols[j]:
                    video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
                    short_title = (
                        video["title"][:62]
                        + ("..." if len(video["title"]) > 62 else "")
                    )
                    st.markdown('<div class="video-card">', unsafe_allow_html=True)
                    st.caption(short_title)
                    st.video(video_url)
                    st.markdown(f"[Open on YouTube]({video_url})")
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        search_link = (
            "https://www.youtube.com/results?search_query="
            + requests.utils.quote(primary_query)
        )
        st.info(f"[Search farming tutorials on YouTube]({search_link})")


CROP_IMAGE_MAP = {
    "apple": "images/apple.jpg",
    "banana": "images/banana.jpg",
    "blackgram": "images/blackgram.jpg",
    "chickpea": "images/chickpea.jpg",
    "coconut": "images/coconut.jpg",
    "coffee": "images/coffee.jpg",
    "cotton": "images/cotton.jpg",
    "grapes": "images/grapes.jpg",
    "jute": "images/jute.jpg",
    "kidneybeans": "images/kidneybeans.jpg",
    "lentil": "images/lentil.jpg",
    "maize": "images/maize.jpg",
    "mango": "images/mango.jpg",
    "mothbeans": "images/mothbeans.jpg",
    "mungbean": "images/mungbeans.jpg",
    "muskmelon": "images/muskmelon.jpg",
    "orange": "images/orange.jpg",
    "papaya": "images/papaya.jpg",
    "pomegranate": "images/pomogranate.jpg",
    "pigeonpeas": "images/pigeonpeas.jpg",
    "rice": "images/rice.jpg",
    "watermelon": "images/watermelon.jpg",
}

CROP_TIPS = {
    "rice": "Maintain 2-5 cm standing water during vegetative growth and split Nitrogen into 3 doses (basal, tillering, panicle initiation).",
    "maize": "Ensure well-drained loamy soil; apply Zinc Sulfate if leaf striping appears and avoid waterlogging at knee-high stage.",
    "chickpea": "Requires minimal irrigation; avoid excess Nitrogen as Rhizobium nodules fix atmospheric nitrogen naturally.",
    "kidneybeans": "Sensitive to frost and waterlogging; maintain moderate soil moisture during flowering and pod filling.",
    "pigeonpeas": "Deep-rooted drought-tolerant pulse; ideal for intercropping with cereals in rainfed regions.",
    "mothbeans": "Thrives in arid and sandy soils with low rainfall; minimal fertilizer input needed.",
    "mungbean": "Short-duration Kharif or summer pulse; apply Phosphorus at sowing to boost root nodulation.",
    "blackgram": "Prefers well-drained black or alluvial soils; monitor for yellow mosaic virus early.",
    "lentil": "Cool-season Rabi crop; avoid excessive irrigation during pod maturation.",
    "pomegranate": "Drip irrigation with mulching improves fruit size and prevents cracking in semi-arid climates.",
    "banana": "High Potassium and moisture feeder; use windbreaks and regular fertigation for uniform bunch weight.",
    "mango": "Withhold irrigation 2-3 months before flowering to induce healthy panicle emergence.",
    "grapes": "Prune strategically and manage canopy humidity to prevent downy and powdery mildew.",
    "watermelon": "Sandy loam soil with warm temperatures yields sweetest fruit; reduce watering near harvest.",
    "muskmelon": "Requires warm, dry weather during ripening for high sugar accumulation.",
    "apple": "Requires winter chilling hours and well-drained slightly acidic soil (pH 5.8-6.8).",
    "orange": "Ensure micronutrient sprays (Zn, Mn, Fe) and avoid water stagnation around the trunk.",
    "papaya": "Strictly requires well-drained soil; even 24 hours of waterlogging can cause root rot.",
    "coconut": "Responds well to organic manure, Potash, and Boron in humid coastal regions.",
    "cotton": "Black cotton soil is ideal; balance N-P-K to prevent excessive vegetative growth over boll formation.",
    "jute": "Requires warm, humid climate with high rainfall and alluvial soil; harvest at small-pod stage for fine fiber.",
    "coffee": "Grows best under filtered shade in high-rainfall hilly terrain with rich organic acidic soil.",
}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0
if "last_recommended_crop" not in st.session_state:
    st.session_state.last_recommended_crop = None
if "last_weather_summary" not in st.session_state:
    st.session_state.last_weather_summary = None
if "crop_result" not in st.session_state:
    st.session_state.crop_result = None
if "yield_result" not in st.session_state:
    st.session_state.yield_result = None

with st.sidebar:
    st.title("Smart Agriculture")
    st.caption("Crop Recommendation and Yield Prediction")
    st.divider()

    nav_options = [
        "Home",
        "Crop Recommendations",
        "Yield Prediction",
        "Weather Forecasting",
        "ChatBot",
    ]

    selected = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state.nav_page),
        label_visibility="collapsed",
    )
    st.session_state.nav_page = selected

    st.divider()
    st.caption("Developed by Cheruku Bhanu Prakash\nIIT Patna - MCA 3rd Sem Project")


# 1. Home Page
if selected == "Home":
    st.markdown(
        """
        <div class="header-box">
            <div class="header-tag">IIT PATNA • MCA 3RD SEMESTER PROJECT</div>
            <h1>Smart Agriculture System</h1>
            <p>
                Machine learning and weather analytics application for crop suitability
                recommendation, regional yield estimation, and farming guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Supported Crops</div>
                <p class="stat-value">22</p>
                <div class="stat-note">Cereals, Pulses & Cash Crops</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Dataset Records</div>
                <p class="stat-value">2,200</p>
                <div class="stat-note">7 Soil & Climate Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">State Coverage</div>
                <p class="stat-value">32</p>
                <div class="stat-note">7 Soil Types Supported</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">ML Algorithm</div>
                <p class="stat-value">RF</p>
                <div class="stat-note">Random Forest Models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    slider_col, info_col = st.columns([1.25, 1], gap="large")

    with slider_col:
        slides = get_home_slides()
        if slides:
            total_slides = len(slides)
            st.session_state.slide_idx %= total_slides
            current_img, current_caption = slides[st.session_state.slide_idx]

            st.image(current_img, use_container_width=True)
            st.markdown(
                f"""
                <div class="slide-caption">
                    <strong>Slide {st.session_state.slide_idx + 1} of {total_slides}:</strong> {current_caption}
                </div>
                """,
                unsafe_allow_html=True,
            )

            s_prev, s_dots, s_next = st.columns([1, 2, 1])
            with s_prev:
                if st.button("< Previous", key="prev_slide", use_container_width=True):
                    st.session_state.slide_idx = (st.session_state.slide_idx - 1) % total_slides
                    st.rerun()
            with s_dots:
                dots = "  ".join(
                    "●" if idx == st.session_state.slide_idx else "○"
                    for idx in range(total_slides)
                )
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;color:#2D6A4F;font-size:1.1rem;'>{dots}</div>",
                    unsafe_allow_html=True,
                )
            with s_next:
                if st.button("Next >", key="next_slide", use_container_width=True):
                    st.session_state.slide_idx = (st.session_state.slide_idx + 1) % total_slides
                    st.rerun()

    with info_col:
        st.subheader("Project Modules")
        st.write("Choose a module below or navigate using the sidebar:")

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">Crop Recommendations</div>
                    <p class="info-card-text">Predicts the best crop from N-P-K, pH, temperature, humidity, and rainfall.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Crop Recommendations", use_container_width=True):
                st.session_state.nav_page = "Crop Recommendations"
                st.rerun()

        with c_b:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">Yield Prediction</div>
                    <p class="info-card-text">Estimates harvest output and cost-profit breakdown by state, soil, and area.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Yield Prediction", use_container_width=True):
                st.session_state.nav_page = "Yield Prediction"
                st.rerun()

        st.write("")
        c_c, c_d = st.columns(2)
        with c_c:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">Weather Forecasting</div>
                    <p class="info-card-text">Displays current weather metrics, 5-day temperature trends, and field advisories.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Weather Forecast", use_container_width=True):
                st.session_state.nav_page = "Weather Forecasting"
                st.rerun()

        with c_d:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">Agriculture ChatBot</div>
                    <p class="info-card-text">Interactive Q&A assistant for fertilizers, soil health, irrigation, and pest control.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open ChatBot", use_container_width=True):
                st.session_state.nav_page = "ChatBot"
                st.rerun()

    with st.expander("Dataset Distribution by Crop (Average Nutrient & Climate Values)"):
        df_home = load_crop_recommendation_data()
        avg_df = (
            df_home.groupby("label")[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
            .mean()
            .round(1)
            .reset_index()
        )
        metric_choice = st.selectbox(
            "Select parameter to compare across crops:",
            ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            format_func=lambda col: {
                "N": "Nitrogen (N)",
                "P": "Phosphorus (P)",
                "K": "Potassium (K)",
                "temperature": "Temperature (°C)",
                "humidity": "Humidity (%)",
                "ph": "Soil pH",
                "rainfall": "Rainfall (mm)",
            }[col],
        )
        fig_home = px.bar(
            avg_df.sort_values(metric_choice, ascending=False),
            x="label",
            y=metric_choice,
            color=metric_choice,
            color_continuous_scale="Greens",
            labels={"label": "Crop", metric_choice: f"Average {metric_choice}"},
            title=f"Average {metric_choice.capitalize()} Requirement by Crop",
        )
        fig_home.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig_home, use_container_width=True)


# 2. Crop Recommendations Page
elif selected == "Crop Recommendations":
    st.markdown(
        """
        <div class="page-heading">
            <h2>Crop Recommendation</h2>
            <p>Adjust soil nutrient sliders and weather inputs for live suitability scoring and benchmark comparison.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rf_model, df = load_crop_recommendation_model()

    st.markdown("**Sample Soil Profiles:**")
    p1, p2, p3, p4 = st.columns(4)
    presets = {
        "rice": (90, 42, 43, 21.0, 82.0, 6.5, 203.0),
        "cotton": (118, 46, 20, 24.0, 80.0, 6.9, 80.0),
        "chickpea": (40, 68, 80, 18.8, 16.7, 7.3, 80.0),
        "coffee": (105, 28, 30, 25.5, 58.0, 6.8, 158.0),
    }

    if "crop_inputs" not in st.session_state:
        st.session_state.crop_inputs = presets["rice"]

    if p1.button("Paddy Soil Sample", use_container_width=True):
        st.session_state.crop_inputs = presets["rice"]
    if p2.button("Cotton Soil Sample", use_container_width=True):
        st.session_state.crop_inputs = presets["cotton"]
    if p3.button("Pulse Soil Sample", use_container_width=True):
        st.session_state.crop_inputs = presets["chickpea"]
    if p4.button("Coffee Soil Sample", use_container_width=True):
        st.session_state.crop_inputs = presets["coffee"]

    def_n, def_p, def_k, def_temp, def_hum, def_ph, def_rain = st.session_state.crop_inputs

    st.markdown("#### Soil Macronutrients (kg/ha)")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_val = st.slider("Nitrogen (N)", min_value=0, max_value=140, value=int(def_n))
    with col2:
        p_val = st.slider("Phosphorus (P)", min_value=0, max_value=145, value=int(def_p))
    with col3:
        k_val = st.slider("Potassium (K)", min_value=0, max_value=205, value=int(def_k))

    st.markdown("#### Climate and Soil pH")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        temp_val = st.slider(
            "Temperature (°C)", min_value=0.0, max_value=55.0, value=float(def_temp), step=0.5
        )
    with c2:
        hum_val = st.slider(
            "Humidity (%)", min_value=0.0, max_value=100.0, value=float(def_hum), step=1.0
        )
    with c3:
        ph_val = st.slider(
            "Soil pH (0-14)", min_value=0.0, max_value=14.0, value=float(def_ph), step=0.1
        )
    with c4:
        rain_val = st.slider(
            "Rainfall (mm)", min_value=0.0, max_value=400.0, value=float(def_rain), step=5.0
        )

    live_mode = st.toggle("Live update as sliders change", value=True)
    predict_clicked = False
    if not live_mode:
        predict_clicked = st.button(
            "Predict Recommended Crop", type="primary", use_container_width=True
        )

    if live_mode or predict_clicked or st.session_state.crop_result is not None:
        if live_mode or predict_clicked:
            input_df = pd.DataFrame(
                [[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]],
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            )
            raw_pred = rf_model.predict(input_df)[0]

            if np.issubdtype(rf_model.classes_.dtype, np.number):
                le = LabelEncoder().fit(df["label"])
                prediction = str(le.inverse_transform([int(raw_pred)])[0])
                class_names = [str(c) for c in le.inverse_transform(rf_model.classes_.astype(int))]
            else:
                prediction = str(raw_pred)
                class_names = [str(c) for c in rf_model.classes_]

            probs = rf_model.predict_proba(input_df)[0]
            top_indices = np.argsort(probs)[::-1][:3]
            top_crops = [(class_names[i], round(probs[i] * 100, 1)) for i in top_indices]

            st.session_state.last_recommended_crop = prediction
            st.session_state.crop_result = {
                "prediction": prediction,
                "top_crops": top_crops,
                "inputs": [n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val],
            }

        res = st.session_state.crop_result
        prediction = res["prediction"]
        top_crops = res["top_crops"]
        n_v, p_v, k_v, temp_v, hum_v, ph_v, rain_v = res["inputs"]

        st.write("")
        res_left, res_right = st.columns([1.1, 1.3], gap="large")

        with res_left:
            st.markdown(
                f"""
                <div class="result-box">
                    <div style="font-size:0.82rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">
                        Recommended Crop • Confidence: {top_crops[0][1]}%
                    </div>
                    <h3>{prediction.capitalize()}</h3>
                    <p>{CROP_TIPS.get(prediction.lower(), 'Suitable for the entered soil and weather parameters.')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            img_path = CROP_IMAGE_MAP.get(prediction.lower(), f"images/{prediction.lower()}.jpg")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Recommended Crop: {prediction.capitalize()}", use_container_width=True)

            prob_df = pd.DataFrame(top_crops, columns=["Crop", "Confidence (%)"])
            prob_df["Crop"] = prob_df["Crop"].str.capitalize()
            fig_prob = px.bar(
                prob_df,
                x="Confidence (%)",
                y="Crop",
                orientation="h",
                text="Confidence (%)",
                color="Confidence (%)",
                color_continuous_scale="Greens",
                title="Top 3 Candidate Crops",
            )
            fig_prob.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                coloraxis_showscale=False,
                height=230,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with res_right:
            crop_subset = df[df["label"].str.lower() == prediction.lower()]
            ideal_means = crop_subset[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].mean()

            categories = ["Nitrogen", "Phosphorus", "Potassium", "Temp (°C)", "Humidity (%)", "pH (x10)", "Rainfall (/2)"]
            user_scaled = [n_v, p_v, k_v, temp_v, hum_v, ph_v * 10, rain_v / 2]
            ideal_scaled = [
                ideal_means["N"],
                ideal_means["P"],
                ideal_means["K"],
                ideal_means["temperature"],
                ideal_means["humidity"],
                ideal_means["ph"] * 10,
                ideal_means["rainfall"] / 2,
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=user_scaled,
                    theta=categories,
                    fill="toself",
                    name="Entered Values",
                    line_color="#D97706",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=ideal_scaled,
                    theta=categories,
                    fill="toself",
                    name=f"{prediction.capitalize()} Dataset Average",
                    line_color="#2D6A4F",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=f"Input Comparison vs. {prediction.capitalize()} Average",
                height=420,
                margin=dict(l=40, r=40, t=50, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            comp_df = pd.DataFrame(
                {
                    "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature (°C)", "Humidity (%)", "Soil pH", "Rainfall (mm)"],
                    "Your Input": [n_v, p_v, k_v, temp_v, hum_v, ph_v, rain_v],
                    f"{prediction.capitalize()} Average": [
                        round(ideal_means["N"], 1),
                        round(ideal_means["P"], 1),
                        round(ideal_means["K"], 1),
                        round(ideal_means["temperature"], 1),
                        round(ideal_means["humidity"], 1),
                        round(ideal_means["ph"], 2),
                        round(ideal_means["rainfall"], 1),
                    ],
                }
            )
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        st.divider()
        render_result_videos(
            title=f"{prediction.capitalize()} Cultivation and Fertilizer Videos",
            primary_query=f"{prediction} crop cultivation and fertilizer management guide",
            fallback_query=f"{prediction} farming guide",
            crop_key=prediction,
            max_results=4,
        )


# 3. Yield Prediction Page
elif selected == "Yield Prediction":
    st.markdown(
        """
        <div class="page-heading">
            <h2>Crop Yield Prediction</h2>
            <p>Adjust area, nutrients, and weather sliders for live yield estimation, cost analysis, and fertilizer guides.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_yield = load_yield_model()
    _, x_columns = load_yield_data()

    states = (
        "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam",
        "Bihar", "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli", "Daman and Diu",
        "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
        "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan",
        "Sikkim", "Tamil Nadu", "Tripura", "Uttar Pradesh",
    )
    soils = ("Alluvial", "Black", "Clayey", "Loamy", "Red", "Sandy", "Silt")
    crops = [
        "Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy",
        "Pulses", "Sugarcane", "Tobacco", "Wheat", "coffee", "kidneybeans", "orange",
        "pomegranate", "rice", "watermelon",
    ]

    default_crop_idx = 6
    if st.session_state.last_recommended_crop:
        rec_lower = str(st.session_state.last_recommended_crop).lower()
        for idx, c in enumerate(crops):
            if c.lower() == rec_lower:
                default_crop_idx = idx
                break

    st.markdown("#### Region and Crop Details")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        state_input = st.selectbox("Select State", states, index=1)
    with r2:
        soil_type = st.selectbox("Select Soil Type", soils)
    with r3:
        crop_name = st.selectbox("Select Crop", crops, index=default_crop_idx)
    with r4:
        area = st.slider("Area (Hectares)", min_value=0.1, max_value=100.0, value=2.0, step=0.5)

    st.markdown("#### Soil Nutrients and Weather Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_soil = st.slider("Nitrogen (N)", min_value=0, max_value=126, value=40)
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0, step=0.5)
    with c2:
        p_soil = st.slider("Phosphorus (P)", min_value=0, max_value=54, value=25)
        humd = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    with c3:
        k_soil = st.slider("Potassium (K)", min_value=0, max_value=59, value=25)
        rain = st.slider("Rainfall (mm)", min_value=0.0, max_value=300.0, value=120.0, step=5.0)
    with c4:
        ph_soil = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.8, step=0.1)

    live_yield = st.toggle("Live yield recalculation on slider change", value=True, key="live_yield_toggle")
    predict_yield_btn = False
    if not live_yield:
        predict_yield_btn = st.button(
            "Predict Crop Yield", type="primary", use_container_width=True
        )

    if live_yield or predict_yield_btn or st.session_state.yield_result is not None:
        if live_yield or predict_yield_btn:
            new_data = pd.DataFrame(
                {
                    "STATE": [state_input],
                    "SOIL_TYPE": [soil_type],
                    "N_SOIL": [n_soil],
                    "P_SOIL": [p_soil],
                    "K_SOIL": [k_soil],
                    "TEMPERATURE": [temp],
                    "HUMIDITY": [humd],
                    "ph": [ph_soil],
                    "RAINFALL": [rain],
                }
            )
            new_data_encoded = (
                pd.get_dummies(new_data)
                .reindex(columns=x_columns, fill_value=0)
                .astype(float)
            )
            predicted_price = float(np.asarray(model_yield.predict(new_data_encoded)).reshape(-1)[0])

            profit = 0.3 * predicted_price
            total_cost = 0.7 * predicted_price
            yi = (profit + total_cost) / predicted_price if predicted_price else 0.0
            yid_total = (yi / 100.0) * area
            yid_per_ha = yi / 100.0

            st.session_state.yield_result = {
                "state_input": state_input,
                "soil_type": soil_type,
                "crop_name": crop_name,
                "area": area,
                "predicted_price": predicted_price,
                "profit": profit,
                "total_cost": total_cost,
                "yid_total": yid_total,
                "yid_per_ha": yid_per_ha,
            }

        yr = st.session_state.yield_result
        state_input = yr["state_input"]
        soil_type = yr["soil_type"]
        crop_name = yr["crop_name"]
        area = yr["area"]
        predicted_price = yr["predicted_price"]
        profit = yr["profit"]
        total_cost = yr["total_cost"]
        yid_total = yr["yid_total"]
        yid_per_ha = yr["yid_per_ha"]

        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Total Expected Yield</div>
                    <p class="stat-value">{yid_total:.1f} qtl</p>
                    <div class="stat-note">For {area:.1f} Hectares ({crop_name})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Yield Per Hectare</div>
                    <p class="stat-value">{yid_per_ha:.1f} qtl/ha</p>
                    <div class="stat-note">{soil_type} Soil • {state_input}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Estimated Crop Price</div>
                    <p class="stat-value">₹{predicted_price:,.0f}</p>
                    <div class="stat-note">Rate per Quintal</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Estimated Net Profit</div>
                    <p class="stat-value">₹{profit * area:,.0f}</p>
                    <div class="stat-note">Across {area:.1f} Hectares</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        chart_col, guide_col = st.columns([1.1, 1.3], gap="large")

        with chart_col:
            econ_df = pd.DataFrame(
                {
                    "Component": ["Cultivation Cost (70%)", "Net Profit (30%)"],
                    "Amount (₹)": [round(total_cost * area, 2), round(profit * area, 2)],
                }
            )
            fig_pie = px.pie(
                econ_df,
                names="Component",
                values="Amount (₹)",
                hole=0.45,
                color_discrete_sequence=["#94A3B8", "#2D6A4F"],
                title=f"Cost vs. Profit Estimate ({area:.1f} ha)",
            )
            fig_pie.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with guide_col:
            if yid_per_ha < 25:
                result_query = f"how to increase {crop_name} yield in {soil_type} soil fertilizer management"
            else:
                result_query = f"{crop_name} farming in {soil_type} soil fertilizer schedule"

            render_result_videos(
                title=f"{crop_name.capitalize()} ({soil_type} Soil) Fertilizer Videos",
                primary_query=result_query,
                fallback_query=f"{crop_name} fertilizers farming guide",
                crop_key=crop_name,
                max_results=4,
            )


# 4. Weather Forecasting Page
elif selected == "Weather Forecasting":
    st.markdown(
        """
        <div class="page-heading">
            <h2>Weather Forecasting</h2>
            <p>Check current temperature, humidity, wind speed, and a 5-day forecast for your city.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Quick City Selection:**")
    q1, q2, q3, q4, q5 = st.columns(5)
    if "weather_city" not in st.session_state:
        st.session_state.weather_city = "Vijayawada"

    if q1.button("Vijayawada", use_container_width=True):
        st.session_state.weather_city = "Vijayawada"
    if q2.button("Hyderabad", use_container_width=True):
        st.session_state.weather_city = "Hyderabad"
    if q3.button("Patna", use_container_width=True):
        st.session_state.weather_city = "Patna"
    if q4.button("Bengaluru", use_container_width=True):
        st.session_state.weather_city = "Bengaluru"
    if q5.button("New Delhi", use_container_width=True):
        st.session_state.weather_city = "New Delhi"

    with st.form("weather_form"):
        in_col, btn_col = st.columns([4, 1])
        with in_col:
            city_name = st.text_input(
                "Enter City Name",
                value=st.session_state.weather_city,
                placeholder="Enter city name...",
                label_visibility="collapsed",
            )
        with btn_col:
            fetch_weather = st.form_submit_button("Get Forecast", type="primary", use_container_width=True)

    if fetch_weather and city_name.strip():
        st.session_state.weather_city = city_name.strip()
        if not OPENWEATHER_API_KEY:
            st.error("Weather forecasting is not configured. Add OPENWEATHER_API_KEY to Streamlit Secrets.")
            st.stop()

        try:
            owm = pyowm.OWM(OPENWEATHER_API_KEY)
            mgr = owm.weather_manager()
            observation = mgr.weather_at_place(city_name.strip())
            weather = observation.weather

            temp_now = weather.temperature("celsius")["temp"]
            humidity_now = weather.humidity
            wind_speed = weather.wind().get("speed", 0)
            pressure_now = weather.pressure.get("press", 0)
            status_desc = weather.detailed_status.title()
            today_str = time.strftime("%d %b %Y", time.localtime())

            forecaster = mgr.forecast_at_place(city_name.strip(), "3h", limit=40)
            forecast_temps = []
            forecast_humidity = []
            for idx, item in enumerate(forecaster.forecast, start=1):
                if idx in (8, 16, 24, 32, 40):
                    forecast_temps.append(item.temperature("celsius")["temp"])
                    forecast_humidity.append(item.humidity)

            weekday_names = []
            now_ts = time.time()
            for _ in range(len(forecast_temps)):
                now_ts += 86400
                weekday_names.append(time.strftime("%a (%d %b)", time.localtime(now_ts)))

            st.session_state.last_weather_summary = (
                f"{city_name.strip()}: {temp_now}°C, {status_desc}, Humidity {humidity_now}%, Wind {wind_speed} m/s"
            )

            st.markdown(
                f"""
                <div class="weather-box">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                        <div>
                            <p style="margin:0;opacity:0.85;font-size:0.9rem;">{today_str} • Current Weather</p>
                            <h2 style="margin:0.2rem 0;font-size:2rem;">{city_name.strip().title()}</h2>
                            <p style="margin:0;font-size:1.05rem;opacity:0.95;">Condition: {status_desc}</p>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:3rem;font-weight:700;line-height:1;">{temp_now:.1f}°C</div>
                            <p style="margin:0.4rem 0 0 0;opacity:0.85;">Observed Temperature</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            w1, w2, w3 = st.columns(3)
            with w1:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">Humidity</div>
                        <p class="stat-value">{humidity_now}%</p>
                        <div class="stat-note">{'High fungal risk' if humidity_now > 80 else 'Normal moisture level'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with w2:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">Wind Speed</div>
                        <p class="stat-value">{wind_speed} m/s</p>
                        <div class="stat-note">{'Avoid foliar spraying' if wind_speed > 5.5 else 'Suitable for spraying'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with w3:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">Pressure</div>
                        <p class="stat-value">{pressure_now} hPa</p>
                        <div class="stat-note">Atmospheric Pressure</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            if forecast_temps:
                fc_df = pd.DataFrame(
                    {
                        "Day": weekday_names,
                        "Temperature (°C)": forecast_temps,
                        "Humidity (%)": forecast_humidity,
                    }
                )
                fig_weather = px.line(
                    fc_df,
                    x="Day",
                    y="Temperature (°C)",
                    markers=True,
                    text="Temperature (°C)",
                    title=f"5-Day Temperature Forecast - {city_name.strip().title()}",
                )
                fig_weather.update_traces(
                    textposition="top center",
                    line=dict(color="#2B5876", width=3),
                    marker=dict(size=9, color="#1B4332"),
                )
                fig_weather.update_layout(
                    height=340,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_weather, use_container_width=True)

            advisories = []
            if temp_now > 35:
                advisories.append("**Heat Advisory:** Schedule irrigation during early morning or late evening to reduce evaporation loss.")
            elif temp_now < 12:
                advisories.append("**Cold Advisory:** Protect young nursery seedlings from low-temperature stress.")
            else:
                advisories.append("**Temperature Status:** Favorable temperature range for crop growth.")

            if wind_speed > 5.5:
                advisories.append("**Spraying Caution:** High wind speeds can cause spray drift; delay foliar spraying.")
            else:
                advisories.append("**Spraying Status:** Calm wind conditions suitable for foliar nutrient or pesticide application.")

            st.info("\n\n".join(advisories))

        except NotFoundError:
            st.error(f"City '{city_name.strip()}' was not found. Please check the spelling.")
        except UnauthorizedError:
            st.error("Invalid OPENWEATHER_API_KEY. Check your key in Streamlit Cloud Secrets.")
        except Exception as e:
            st.error(f"Weather service error: {e}")


# 5. Agriculture ChatBot Page
elif selected == "ChatBot":
    top_col, clear_col = st.columns([5, 1])
    with top_col:
        st.markdown(
            """
            <div class="page-heading">
                <h2>Agriculture Assistant</h2>
                <p>Ask questions about crops, soil health, fertilizers, irrigation, and pest management.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with clear_col:
        st.write("")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not GROQ_API_KEY:
        st.error(
            "Groq API key is missing. Add GROQ_API_KEY in "
            "Streamlit Cloud -> Manage app -> Settings -> Secrets."
        )
        st.stop()

    client = Groq(api_key=GROQ_API_KEY)
    MODEL_NAME = "openai/gpt-oss-120b"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    context_notes = []
    if st.session_state.last_recommended_crop is not None:
        crop_label = str(st.session_state.last_recommended_crop).capitalize()
        context_notes.append(f"Recommended Crop: {crop_label}")
    if st.session_state.last_weather_summary:
        context_notes.append(f"Weather: {st.session_state.last_weather_summary}")
    if context_notes:
        st.caption("Session Context: " + " | ".join(context_notes))

    starter_prompt = None
    if not st.session_state.messages:
        st.markdown("**Sample Questions:**")
        s1, s2, s3 = st.columns(3)
        if s1.button("Fertilizer schedule for Kharif paddy", use_container_width=True):
            starter_prompt = "What is the recommended NPK fertilizer schedule for Kharif paddy?"
        if s2.button("How to treat acidic soil (pH 5.2)", use_container_width=True):
            starter_prompt = "How can a farmer reclaim acidic soil with pH 5.2 before sowing?"
        if s3.button("Drip irrigation schedule for cotton", use_container_width=True):
            starter_prompt = "Give a practical drip irrigation and fertigation schedule for cotton."

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about crops, farming, soil, fertilizers...") or starter_prompt

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    system_prompt = (
                        "You are an agricultural assistant. Give clear, practical, "
                        "and responsible guidance about crops, soil health, fertilizers, "
                        "irrigation, pests, and farming. Do not invent laboratory results "
                        "or guarantee yields. Ask for missing details when needed and "
                        "mention uncertainty where appropriate."
                    )
                    if context_notes:
                        system_prompt += f" User's current session context: {'; '.join(context_notes)}."

                    conversation = [{"role": "system", "content": system_prompt}]
                    conversation.extend(
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-12:]
                    )

                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=conversation,
                        temperature=0.5,
                        max_completion_tokens=1024,
                    )
                    answer = (
                        completion.choices[0].message.content
                        or "No response was generated. Please try again."
                    )
                except Exception as e:
                    error_message = str(e)
                    if "401" in error_message or "authentication" in error_message.lower():
                        answer = "Groq authentication failed. Check that GROQ_API_KEY is valid."
                    elif "429" in error_message or "rate_limit" in error_message.lower():
                        answer = "Groq rate limit reached. Please wait a moment and try again."
                    else:
                        answer = "Unable to generate a response right now. Please try again shortly."

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown(
    """
    <div class="footer-text">
        Smart Agriculture System • All Rights Reserved 2026
        <br/>Developed by Cheruku Bhanu Prakash | IIT Patna MCA Project
    </div>
    """,
    unsafe_allow_html=True,
)
