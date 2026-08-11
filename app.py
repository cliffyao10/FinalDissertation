import html
import io

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.recognition import (
    extract_image_embedding,
    predict_category_with_confidence,
    predict_semantic_colour,
    predict_style,
)
from src.colour_detection import predict_colour_details
from src.recommendation import recommend_outfit
from src.weather import WeatherServiceError, get_city_weather


FRAME_WIDTH = 520
FRAME_HEIGHT = 400
FRAME_BACKGROUND = (250, 248, 244)

# Product mode shows recommendations only. Change this to True while
# collecting dissertation evidence or diagnosing recognition failures.
DEVELOPER_MODE = False

STYLE_OPTIONS = [
    "Casual",
    "Outdoor",
    "Sporty",
    "Formal",
    "Business",
    "Streetwear",
    "Minimalist",
    "Party",
    "Beachwear",
]

ICON_COLOURS = {
    "Black": "#26272b",
    "White": "#f7f7f5",
    "Grey": "#8b9098",
    "Blue": "#527db5",
    "Navy": "#293f62",
    "Red": "#bd4b51",
    "Green": "#66846a",
    "Olive": "#77744e",
    "Brown": "#87654e",
    "Beige": "#d7c4a5",
    "Cream": "#eee2c5",
    "Purple": "#806591",
    "Pink": "#d99aaa",
    "Orange": "#d4864a",
    "Yellow": "#d8ba4c",
}

THEMES = {
    "Blue": {
        "primary": "#6486a7", "strong": "#3f6385", "soft": "#e8f0f7",
        "glow": "#abc9e3", "secondary": "#d8e3ee", "ink": "#172b3d",
        "muted": "#617486", "surface": "#f8fbfd", "canvas": "#3f6385",
        "canvas_fill": "rgba(63, 99, 133, 0.12)",
    },
    "Green": {
        "primary": "#789986", "strong": "#456b55", "soft": "#e9f2ec",
        "glow": "#9fc9ad", "secondary": "#e3d1bc", "ink": "#17251e",
        "muted": "#657069", "surface": "#f8faf8", "canvas": "#456b55",
        "canvas_fill": "rgba(69, 107, 85, 0.12)",
    },
    "Pink": {
        "primary": "#c08699", "strong": "#995d72", "soft": "#f6e9ee",
        "glow": "#e6b7c6", "secondary": "#efd8cc", "ink": "#38242c",
        "muted": "#7a6870", "surface": "#fdf9fa", "canvas": "#995d72",
        "canvas_fill": "rgba(153, 93, 114, 0.12)",
    },
    "Beige": {
        "primary": "#a48a6c", "strong": "#775d42", "soft": "#f3ece3",
        "glow": "#dac3a7", "secondary": "#eadbca", "ink": "#33291f",
        "muted": "#766c62", "surface": "#fcfaf7", "canvas": "#775d42",
        "canvas_fill": "rgba(119, 93, 66, 0.12)",
    },
    "Minimal White": {
        "primary": "#8a8a86", "strong": "#b33c45", "soft": "#f1f1ef",
        "glow": "#e7e7e3", "secondary": "#f4f4f1", "ink": "#20201f",
        "muted": "#70706d", "surface": "#ffffff", "canvas": "#b33c45",
        "canvas_fill": "rgba(179, 60, 69, 0.10)",
    },
}

CATEGORY_CHOICES = [
    "Tank Top",
    "T-Shirt",
    "Shirt",
    "Blouse",
    "Sweater",
    "Hoodie",
    "Jacket",
    "Blazer",
    "Coat",
    "Jeans",
    "Trousers",
    "Shorts",
    "Skirt",
    "Dress",
    "Bikini",
    "Swimsuit",
    "Shoes",
]

CATEGORY_RECOMMENDATION_ALIASES = {
    "Tank Top": "T-Shirt",
    "Blouse": "Shirt",
    "Blazer": "Jacket",
}

CATEGORY_PARENTS = {
    "Tank Top": "Top",
    "T-Shirt": "Top",
    "Shirt": "Top",
    "Blouse": "Top",
    "Sweater": "Top",
    "Hoodie": "Top",
    "Jacket": "Outerwear",
    "Blazer": "Outerwear",
    "Coat": "Outerwear",
    "Jeans": "Bottom",
    "Trousers": "Bottom",
    "Shorts": "Bottom",
    "Skirt": "Bottom",
    "Dress": "One-piece",
    "Bikini": "Swimwear",
    "Swimsuit": "Swimwear",
    "Shoes": "Footwear",
}


st.set_page_config(
    page_title="AI Outfit Recommendation",
    page_icon="👕",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }

        h1 {
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
        }

        .result-item {
            min-height: 66px;
            padding: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid #d9eee0;
            border-radius: 12px;
            background: #f3faf5;
            color: #166534;
            font-weight: 600;
        }

        .empty-result {
            height: 400px;
            padding: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px dashed #d1d5db;
            border-radius: 16px;
            background: #fafafa;
            color: #9ca3af;
        }

        .fixed-image-frame {
            width: 100%;
            max-width: 520px;
            height: 400px;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            background: #f6f7f9;
            overflow: hidden;
        }

        .stButton > button {
            min-height: 42px;
            border-radius: 11px;
            font-weight: 600;
        }

        div[data-testid="stFileUploader"] {
            margin-top: 0.35rem;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stFileUploaderDropzone"] {
            min-height: 110px;
            border-radius: 15px;
        }

        div[data-testid="stImage"] {
            max-width: 520px;
        }

        div[data-testid="stImage"] img {
            border-radius: 15px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 15px;
        }

        div[data-testid="stMetric"] {
            padding: 0;
        }
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(94vw, 1450px) !important;
            max-width: 1450px !important;
            height: min(92vh, 920px) !important;
            max-height: 92vh !important;
        }

        div[data-testid="stDialog"] div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        div[data-testid="stDialog"] h2,
        div[data-testid="stDialog"] h3,
        div[data-testid="stDialog"] h4 {
            margin-top: 0.25rem !important;
            margin-bottom: 0.35rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Shared visual language for the opening screen, workspace and weather card.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.45rem;
            padding-bottom: 2.5rem;
            position: relative;
            z-index: 1;
        }

        h1 {
            font-size: clamp(2rem, 3.1vw, 2.75rem) !important;
            letter-spacing: -0.045em;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(220, 236, 228, .62), transparent 31rem),
                radial-gradient(circle at 90% 20%, rgba(236, 222, 205, .44), transparent 28rem),
                #f8faf8;
        }

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            width: 28rem;
            height: 28rem;
            border-radius: 999px;
            filter: blur(80px);
            opacity: .22;
            pointer-events: none;
            z-index: 0;
            animation: ambient-light 14s ease-in-out infinite alternate;
        }

        .stApp::before {
            top: 4%;
            left: -10rem;
            background: #9fc9ad;
        }

        .stApp::after {
            right: -8rem;
            bottom: 3%;
            background: #e1b98e;
            animation-delay: -7s;
        }

        @keyframes ambient-light {
            from { transform: translate3d(0, -1rem, 0) scale(.92); }
            to { transform: translate3d(3rem, 2rem, 0) scale(1.08); }
        }

        .app-wordmark {
            display: inline-flex;
            align-items: center;
            gap: .65rem;
            margin-bottom: 1.05rem;
            color: #274737;
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .app-wordmark::before {
            content: "";
            width: 2.15rem;
            height: 1px;
            background: #789986;
        }

        .opening-kicker {
            color: #557565;
            font-size: .7rem;
            font-weight: 750;
            letter-spacing: .2em;
            text-transform: uppercase;
        }

        .opening-title {
            max-width: 540px;
            margin: .75rem 0 .85rem;
            color: #17251e;
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2.1rem, 3vw, 3.25rem);
            font-weight: 500;
            letter-spacing: -.05em;
            line-height: 1.02;
        }

        .opening-subtitle {
            max-width: 465px;
            margin: 0 0 1.5rem;
            color: #657069;
            font-size: clamp(.95rem, 1.35vw, 1.08rem);
            line-height: 1.62;
        }

        .home-wordmark {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: .25rem 0 2rem;
            color: var(--fashion-ink);
        }

        .home-brand {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.25rem;
            letter-spacing: -.035em;
        }

        .home-edition {
            color: var(--fashion-muted);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        .home-greeting {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            margin-top: 1.25rem;
            color: var(--fashion-muted);
            font-size: .75rem;
        }

        .home-greeting::before {
            content: "";
            width: 1.6rem;
            height: 1px;
            background: var(--fashion-primary);
        }

        .weather-window-scene {
            position: relative;
            min-height: 610px;
            overflow: hidden;
            isolation: isolate;
            --sky-top: #abcfe9;
            --sky-middle: #d8e8ed;
            --sky-bottom: #f4dfc4;
            --cloud-front: rgba(255, 255, 255, .96);
            --cloud-back: rgba(222, 235, 241, .9);
            border: 1px solid rgba(255, 255, 255, .76);
            border-radius: 38px 38px 16px 38px;
            background:
                radial-gradient(circle at 77% 18%, rgba(255,255,255,.34), transparent 32%),
                linear-gradient(155deg, var(--sky-top), var(--sky-middle) 52%, var(--sky-bottom));
            box-shadow:
                0 32px 72px rgba(82, 70, 58, .13),
                inset 0 0 0 9px rgba(255, 252, 247, .26);
            transition: background .55s ease;
        }

        .weather-window-scene::before {
            content: "";
            position: absolute;
            inset: 11px;
            z-index: 8;
            border: 1px solid rgba(255,255,255,.3);
            border-radius: 29px 29px 10px 29px;
            box-shadow: inset 0 0 40px rgba(255,255,255,.08);
            pointer-events: none;
        }

        .weather-window-scene::after {
            content: "";
            position: absolute;
            right: -15%;
            bottom: -19%;
            left: -15%;
            z-index: 1;
            height: 48%;
            border-radius: 50% 50% 0 0;
            background:
                radial-gradient(circle at 68% 5%, rgba(255,239,213,.68), transparent 35%),
                linear-gradient(180deg, rgba(255,230,201,.2), rgba(247,220,190,.7));
            filter: blur(2px);
        }

        .weather-cloudy {
            --sky-top: #aebfd3;
            --sky-middle: #d7dce5;
            --sky-bottom: #efd8c7;
            --cloud-front: rgba(244, 247, 250, .96);
            --cloud-back: rgba(181, 197, 211, .9);
        }
        .weather-rain {
            --sky-top: #748ca9;
            --sky-middle: #9eb2c6;
            --sky-bottom: #d8cbd0;
            --cloud-front: rgba(221, 230, 238, .97);
            --cloud-back: rgba(105, 126, 150, .94);
        }
        .weather-snow {
            --sky-top: #b9cfe0;
            --sky-middle: #e1e8ee;
            --sky-bottom: #f2e8dd;
            --cloud-front: rgba(249, 251, 252, .98);
            --cloud-back: rgba(195, 211, 224, .92);
        }
        .weather-fog {
            --sky-top: #bac8ce;
            --sky-middle: #dfe1df;
            --sky-bottom: #ead9ca;
        }
        .weather-thunderstorm {
            --sky-top: #596b88;
            --sky-middle: #7c8da7;
            --sky-bottom: #c4afb7;
            --cloud-front: rgba(190, 202, 215, .98);
            --cloud-back: rgba(76, 92, 118, .96);
        }

        .weather-scene-label {
            position: absolute;
            top: 1.55rem;
            left: 1.75rem;
            z-index: 12;
            display: flex;
            align-items: center;
            gap: .6rem;
            color: rgba(43, 57, 70, .68);
            font-size: .66rem;
            font-weight: 760;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .weather-scene-label::before {
            content: "";
            width: .46rem;
            height: .46rem;
            border: 1px solid currentColor;
            border-radius: 999px;
            box-shadow: inset 0 0 0 2px rgba(255,255,255,.25);
        }
        .weather-rain .weather-scene-label,
        .weather-thunderstorm .weather-scene-label { color: rgba(248,250,255,.75); }

        .weather-illustration {
            position: absolute;
            inset: 2.8rem 1.3rem 6.2rem;
            z-index: 3;
        }

        .weather-orb {
            position: absolute;
            top: 16%;
            right: 13%;
            width: 9.7rem;
            aspect-ratio: 1;
            border-radius: 999px;
            background: linear-gradient(145deg, #fff3b9, #ffd173 72%);
            box-shadow:
                0 0 0 24px rgba(255,235,169,.12),
                0 22px 55px rgba(235,173,75,.24);
            animation: weather-orb-breathe 7s ease-in-out infinite;
        }
        .weather-cloudy .weather-orb { opacity: .46; transform: translate(-2rem, 1rem) scale(.84); }
        .weather-rain .weather-orb,
        .weather-fog .weather-orb,
        .weather-snow .weather-orb,
        .weather-thunderstorm .weather-orb { opacity: 0; }

        .weather-cloud-form {
            position: absolute;
            top: 43%;
            left: 15%;
            width: 58%;
            height: 5.6rem;
            border-radius: 999px;
            background: var(--cloud-front);
            box-shadow: 0 22px 36px rgba(78,101,122,.13);
            animation: weather-cloud-float 7s ease-in-out infinite;
        }
        .weather-cloud-form::before,
        .weather-cloud-form::after {
            content: "";
            position: absolute;
            bottom: 17%;
            border-radius: 999px;
            background: inherit;
        }
        .weather-cloud-form::before {
            left: 16%;
            width: 7.5rem;
            height: 7.5rem;
        }
        .weather-cloud-form::after {
            right: 12%;
            width: 5.2rem;
            height: 5.2rem;
        }
        .weather-clear .weather-cloud-form {
            top: 55%;
            left: 4%;
            width: 42%;
            height: 3.6rem;
            opacity: .77;
            transform: scale(.84);
        }
        .weather-clear .weather-cloud-form::before { width: 5.1rem; height: 5.1rem; }
        .weather-clear .weather-cloud-form::after { width: 3.6rem; height: 3.6rem; }
        .weather-rain .weather-cloud-form,
        .weather-snow .weather-cloud-form,
        .weather-thunderstorm .weather-cloud-form { top: 34%; left: 16%; width: 68%; }
        .weather-fog .weather-cloud-form { opacity: .52; filter: blur(1px); }

        .weather-cloud-shadow {
            position: absolute;
            top: 33%;
            left: 10%;
            width: 50%;
            height: 4.3rem;
            border-radius: 999px;
            background: var(--cloud-back);
            opacity: .72;
            filter: blur(.2px);
            animation: weather-cloud-float 8.5s ease-in-out -2s infinite reverse;
        }
        .weather-clear .weather-cloud-shadow { display: none; }

        .weather-face {
            position: absolute;
            top: 49%;
            left: 38%;
            z-index: 6;
            width: 5.2rem;
            height: 2.5rem;
            color: rgba(69, 83, 97, .78);
            animation: weather-face-bob 7s ease-in-out infinite;
        }
        .weather-face .eye {
            position: absolute;
            top: .4rem;
            width: .48rem;
            height: .6rem;
            border-radius: 999px;
            background: currentColor;
            box-shadow: inset 0 .12rem rgba(255,255,255,.22);
        }
        .weather-face .eye.left { left: 1.2rem; }
        .weather-face .eye.right { right: 1.2rem; }
        .weather-face .mouth {
            position: absolute;
            top: .82rem;
            left: 50%;
            width: .9rem;
            height: .55rem;
            border-bottom: 2px solid currentColor;
            border-radius: 0 0 999px 999px;
            transform: translateX(-50%);
        }
        .weather-face .cheek {
            position: absolute;
            top: 1.22rem;
            width: .72rem;
            height: .35rem;
            border-radius: 50%;
            background: rgba(234, 145, 147, .28);
            filter: blur(.2px);
        }
        .weather-face .cheek.left { left: .35rem; }
        .weather-face .cheek.right { right: .35rem; }
        .weather-clear .weather-face {
            top: 30%;
            right: 17.5%;
            left: auto;
            color: rgba(111, 83, 49, .72);
            transform: scale(1.08);
        }
        .weather-rain .weather-face,
        .weather-snow .weather-face,
        .weather-thunderstorm .weather-face { top: 40%; left: 41%; }
        .weather-rain .weather-face .mouth {
            top: 1.18rem;
            height: .38rem;
            border-top: 2px solid currentColor;
            border-bottom: 0;
            border-radius: 999px 999px 0 0;
        }
        .weather-snow .weather-face .eye {
            height: .35rem;
            border-bottom: 2px solid currentColor;
            background: transparent;
        }
        .weather-thunderstorm .weather-face .mouth {
            top: 1rem;
            width: .48rem;
            height: .48rem;
            border: 2px solid currentColor;
            border-radius: 50%;
        }
        .weather-fog .weather-face { opacity: .58; }

        .weather-sparkles {
            position: absolute;
            inset: 0;
            z-index: 5;
            pointer-events: none;
        }
        .weather-sparkles i {
            position: absolute;
            width: .75rem;
            height: .75rem;
            border-radius: 3px;
            background: rgba(255,255,255,.72);
            clip-path: polygon(50% 0, 62% 38%, 100% 50%, 62% 62%, 50% 100%, 38% 62%, 0 50%, 38% 38%);
            animation: weather-twinkle 3.8s ease-in-out infinite;
        }
        .weather-sparkles i:first-child { top: 23%; left: 19%; }
        .weather-sparkles i:last-child { top: 65%; right: 12%; animation-delay: -1.9s; transform: scale(.65); }
        .weather-rain .weather-sparkles,
        .weather-fog .weather-sparkles,
        .weather-thunderstorm .weather-sparkles { display: none; }

        .weather-precipitation {
            position: absolute;
            top: 58%;
            left: 28%;
            width: 49%;
            height: 8rem;
            opacity: 0;
        }
        .weather-precipitation i {
            position: absolute;
            top: 0;
            left: calc(var(--drop) * 15%);
            width: 3px;
            height: 2.1rem;
            border-radius: 999px;
            background: rgba(224, 241, 255, .9);
            transform: rotate(12deg);
            animation: weather-rain-drop 1.25s ease-in infinite;
            animation-delay: calc(var(--drop) * -.17s);
        }
        .weather-rain .weather-precipitation,
        .weather-thunderstorm .weather-precipitation { opacity: 1; }
        .weather-snow .weather-precipitation { opacity: 1; }
        .weather-snow .weather-precipitation i {
            width: .7rem;
            height: .7rem;
            border-radius: 50%;
            background: rgba(255,255,255,.95);
            box-shadow: 0 0 12px rgba(255,255,255,.72);
            animation: weather-snow-drop 4.8s ease-in infinite;
        }

        .weather-mist {
            position: absolute;
            inset: 38% 8% auto;
            display: none;
            flex-direction: column;
            gap: 1.1rem;
        }
        .weather-fog .weather-mist { display: flex; }
        .weather-mist i {
            height: 1.15rem;
            border-radius: 999px;
            background: rgba(255,255,255,.48);
            backdrop-filter: blur(5px);
            animation: weather-mist-drift 6s ease-in-out infinite alternate;
        }
        .weather-mist i:nth-child(1) { width: 68%; align-self: flex-end; }
        .weather-mist i:nth-child(2) { width: 86%; animation-delay: -2s; }
        .weather-mist i:nth-child(3) { width: 61%; align-self: center; animation-delay: -4s; }

        .weather-bolt {
            display: none;
            content: "";
            position: absolute;
            top: 57%;
            left: 50%;
            width: 2.4rem;
            height: 5rem;
            background: linear-gradient(160deg, #fff0a2, #ffc95f);
            clip-path: polygon(48% 0, 100% 0, 65% 38%, 92% 38%, 18% 100%, 39% 54%, 8% 54%);
            filter: drop-shadow(0 0 13px rgba(255,220,112,.58));
            animation: weather-lightning 7s step-end infinite;
        }
        .weather-thunderstorm .weather-bolt { display: block; }

        .weather-glass-light {
            position: absolute;
            top: -15%;
            right: -8%;
            z-index: 7;
            width: 55%;
            height: 88%;
            border-radius: 50%;
            background: linear-gradient(105deg, transparent 24%, rgba(255,255,255,.2), transparent 68%);
            transform: rotate(-11deg);
            pointer-events: none;
        }

        .weather-window-info {
            position: absolute;
            right: 1.4rem;
            bottom: 1.35rem;
            left: 1.4rem;
            z-index: 12;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.2rem;
            min-height: 4.65rem;
            padding: .9rem 1.15rem;
            border: 1px solid rgba(255,255,255,.6);
            border-radius: 22px 22px 9px 22px;
            background: rgba(255, 252, 247, .72);
            color: #384551;
            box-shadow: 0 16px 38px rgba(66,69,76,.11);
            backdrop-filter: blur(20px) saturate(1.08);
        }
        .weather-window-copy strong { display: block; font-size: .94rem; font-weight: 720; }
        .weather-window-copy span { display: block; margin-top: .16rem; color: rgba(56,69,81,.66); font-size: .7rem; }
        .weather-window-temperature {
            font-size: 2rem;
            font-weight: 570;
            line-height: 1;
            letter-spacing: -.06em;
            white-space: nowrap;
        }

        @keyframes weather-orb-breathe {
            50% { transform: scale(1.035); box-shadow: 0 0 0 31px rgba(255,235,169,.08), 0 24px 62px rgba(235,173,75,.28); }
        }
        @keyframes weather-cloud-float { 50% { transform: translateY(-7px); } }
        @keyframes weather-face-bob { 50% { margin-top: -7px; } }
        @keyframes weather-twinkle { 50% { opacity: .28; transform: rotate(28deg) scale(.72); } }
        @keyframes weather-rain-drop {
            0% { transform: translate(8px, -18px) rotate(12deg); opacity: 0; }
            18% { opacity: .9; }
            100% { transform: translate(-10px, 96px) rotate(12deg); opacity: 0; }
        }
        @keyframes weather-snow-drop {
            0% { transform: translate(0, -12px); opacity: 0; }
            16% { opacity: .95; }
            100% { transform: translate(18px, 105px); opacity: 0; }
        }
        @keyframes weather-mist-drift { to { transform: translateX(8%); opacity: .68; } }
        @keyframes weather-lightning { 0%, 91%, 94%, 100% { opacity: .75; } 92%, 93% { opacity: .15; } }
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.city-panel-marker) {
            max-width: 880px;
            margin: 0 auto 1.65rem;
            padding: .45rem .65rem .55rem;
            border: 1px solid rgba(126, 158, 139, .32);
            border-radius: 24px;
            background: rgba(255, 255, 255, .78);
            box-shadow: 0 20px 55px rgba(34, 59, 45, .08);
            backdrop-filter: blur(16px);
        }

        .city-panel-marker {
            display: flex;
            align-items: center;
            gap: .8rem;
        }

        .city-panel-icon {
            display: grid;
            width: 2.6rem;
            height: 2.6rem;
            place-items: center;
            border-radius: 14px;
            background: #e9f2ec;
            font-size: 1.15rem;
        }

        .city-panel-title {
            color: #20372b;
            font-size: 1.03rem;
            font-weight: 720;
        }

        .city-panel-copy {
            color: #7b857f;
            font-size: .82rem;
        }

        .location-pill {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .38rem .7rem;
            border-radius: 999px;
            background: #eef4f0;
            color: #456553;
            font-size: .8rem;
            font-weight: 650;
        }

        .weather-card {
            position: relative;
            overflow: hidden;
            margin-top: 1rem;
            padding: 1.1rem 1.15rem;
            border: 1px solid rgba(129, 155, 139, .30);
            border-radius: 20px;
            background: linear-gradient(145deg, rgba(248,252,249,.96), rgba(232,241,235,.92));
            box-shadow: 0 14px 38px rgba(46, 72, 57, .09);
        }

        .weather-card::after {
            content: "";
            position: absolute;
            top: -4rem;
            right: -3rem;
            width: 10rem;
            height: 10rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255,215,143,.42), transparent 68%);
            animation: weather-glow 6s ease-in-out infinite alternate;
        }

        @keyframes weather-glow {
            from { transform: translate(-.4rem, -.25rem) scale(.9); opacity: .65; }
            to { transform: translate(.6rem, .45rem) scale(1.12); opacity: 1; }
        }

        .weather-head,
        .weather-stats,
        .weather-advice {
            position: relative;
            z-index: 1;
        }

        .weather-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
        }

        .weather-location {
            color: #315240;
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .09em;
            text-transform: uppercase;
        }

        .weather-condition {
            color: #52675b;
            font-size: .82rem;
        }

        .weather-icon {
            font-size: 1.7rem;
            filter: drop-shadow(0 5px 8px rgba(78, 92, 65, .15));
            animation: weather-float 3.4s ease-in-out infinite;
        }

        @keyframes weather-float {
            50% { transform: translateY(-4px); }
        }

        .weather-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .55rem;
            margin-top: .9rem;
        }

        .weather-stat {
            padding: .65rem .55rem;
            border-radius: 13px;
            background: rgba(255,255,255,.62);
        }

        .weather-stat-label {
            color: #849087;
            font-size: .68rem;
        }

        .weather-stat-value {
            margin-top: .08rem;
            color: #263e31;
            font-size: .94rem;
            font-weight: 720;
        }

        .weather-advice {
            margin-top: .8rem;
            padding-top: .75rem;
            border-top: 1px solid rgba(115, 143, 126, .2);
            color: #52675b;
            font-size: .79rem;
            line-height: 1.52;
        }

        .weather-advice strong { color: #284b38; }

        .empty-result {
            background: rgba(255,255,255,.64);
            color: #87918b;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.55);
            backdrop-filter: blur(10px);
        }

        .stButton > button {
            border-radius: 13px;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 9px 24px rgba(32, 60, 44, .10);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 20px;
            background: rgba(255,255,255,.72);
            backdrop-filter: blur(12px);
        }

        @media (prefers-reduced-motion: reduce) {
            .stApp::before,
            .stApp::after,
            .weather-card::after,
            .weather-icon,
            .weather-orb,
            .weather-cloud-form,
            .weather-cloud-shadow,
            .weather-face,
            .weather-sparkles i,
            .weather-precipitation i,
            .weather-mist i,
            .weather-bolt {
                animation: none !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialise_state():
    """Initialise application state."""

    defaults = {
        "image_mode": None,
        "analysis_result": None,
        "selected_image": None,
        "selection_ready": False,
        "canvas_version": 0,
        "uploader_version": 0,
        "uploaded_bytes": None,
        "uploaded_name": None,
        "city": "London",
        "city_input": "London",
        "weather_data": None,
        "weather_error": None,
        "theme": "Green",
    }

    for state_name, default_value in defaults.items():
        if state_name not in st.session_state:
            st.session_state[state_name] = default_value


def render_theme_css():
    """Apply the selected visual palette across every application page."""

    theme = THEMES.get(st.session_state.theme, THEMES["Green"])
    st.markdown(
        f"""
        <style>
            :root {{
                --fashion-primary: {theme["primary"]};
                --fashion-strong: {theme["strong"]};
                --fashion-soft: {theme["soft"]};
                --fashion-glow: {theme["glow"]};
                --fashion-secondary: {theme["secondary"]};
                --fashion-ink: {theme["ink"]};
                --fashion-muted: {theme["muted"]};
                --fashion-surface: {theme["surface"]};
                --fashion-cream: #fff9f3;
                --fashion-blush: #f3dcda;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at 8% 8%, color-mix(in srgb, var(--fashion-glow) 58%, transparent), transparent 31rem),
                    radial-gradient(circle at 90% 20%, color-mix(in srgb, var(--fashion-secondary) 52%, transparent), transparent 28rem),
                    var(--fashion-surface) !important;
                color: var(--fashion-ink);
                font-family: "Segoe UI Variable Text", "Aptos", "Segoe UI", sans-serif;
            }}

            h1, h2, h3, h4,
            .opening-title, .home-brand,
            .weather-window-copy strong {{
                font-family: ui-rounded, "Segoe UI Variable Display", "Trebuchet MS", sans-serif !important;
                font-weight: 680 !important;
                letter-spacing: -.045em;
            }}

            p, label, input, button {{
                letter-spacing: .005em;
            }}

            .stApp::before {{ background: var(--fashion-glow) !important; }}
            .stApp::after {{ background: var(--fashion-secondary) !important; }}
            .app-wordmark, .opening-kicker, .weather-location {{
                color: var(--fashion-strong) !important;
            }}
            .app-wordmark::before {{ background: var(--fashion-primary) !important; }}
            .app-wordmark::after {{
                content: "✦";
                display: grid;
                width: 1.65rem;
                height: 1.65rem;
                margin-left: .1rem;
                place-items: center;
                border-radius: 999px;
                background: color-mix(in srgb, var(--fashion-blush) 78%, white);
                color: var(--fashion-strong);
                font-size: .65rem;
                letter-spacing: 0;
            }}
            .opening-title, .city-panel-title, .weather-stat-value {{
                color: var(--fashion-ink) !important;
            }}
            .opening-subtitle, .city-panel-copy, .weather-condition,
            .weather-advice, .garment-card-colour {{
                color: var(--fashion-muted) !important;
            }}
            .city-panel-icon, .location-pill {{
                background: var(--fashion-soft) !important;
                color: var(--fashion-strong) !important;
            }}
            .weather-card {{
                border-color: color-mix(in srgb, var(--fashion-primary) 35%, transparent) !important;
                border-radius: 28px 28px 14px 28px !important;
                background: linear-gradient(145deg, rgba(255,255,255,.96), var(--fashion-soft)) !important;
                box-shadow: 0 14px 38px color-mix(in srgb, var(--fashion-primary) 15%, transparent) !important;
            }}
            .weather-card::after {{
                background: radial-gradient(circle, color-mix(in srgb, var(--fashion-secondary) 68%, transparent), transparent 68%) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.city-panel-marker) {{
                box-shadow: 0 20px 55px color-mix(in srgb, var(--fashion-primary) 14%, transparent) !important;
            }}
            .weather-advice strong {{ color: var(--fashion-strong) !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"] {{
                border-color: color-mix(in srgb, var(--fashion-primary) 25%, #e7e7e4) !important;
                border-radius: 26px 26px 14px 26px !important;
                background: color-mix(in srgb, var(--fashion-cream) 78%, white) !important;
                box-shadow: 0 14px 38px color-mix(in srgb, var(--fashion-primary) 10%, transparent);
            }}
            .stButton > button:hover,
            .stFormSubmitButton > button:hover {{
                border-color: var(--fashion-primary) !important;
                color: var(--fashion-strong) !important;
            }}
            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {{
                border-color: var(--fashion-strong) !important;
                background: var(--fashion-strong) !important;
                color: white !important;
                box-shadow: 0 8px 22px color-mix(in srgb, var(--fashion-strong) 25%, transparent) !important;
            }}
            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button[kind="primary"]:hover {{
                filter: brightness(.92);
                color: white !important;
            }}
            .stButton > button,
            .stFormSubmitButton > button {{
                border-radius: 999px !important;
            }}
            .garment-card {{
                min-height: 190px;
                padding: 18px 12px;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 20%, #e7e7e4);
                border-radius: 26px 26px 12px 26px;
                background: linear-gradient(150deg, white, color-mix(in srgb, var(--fashion-soft) 46%, var(--fashion-cream)));
                text-align: center;
                box-shadow: 0 11px 28px color-mix(in srgb, var(--fashion-primary) 10%, transparent);
                transition: transform .2s ease, box-shadow .2s ease;
            }}
            .garment-card:hover {{
                transform: translateY(-3px) rotate(-.25deg);
                box-shadow: 0 17px 34px color-mix(in srgb, var(--fashion-primary) 16%, transparent);
            }}
            .garment-icon-shell {{
                display: grid;
                width: 92px;
                height: 92px;
                margin: 0 auto .2rem;
                place-items: center;
                border-radius: 999px;
                background: linear-gradient(145deg, white, color-mix(in srgb, var(--fashion-blush) 42%, var(--fashion-soft)));
                box-shadow: inset 0 0 0 1px rgba(255,255,255,.72);
            }}
            .garment-card-slot {{
                margin-top: 4px;
                color: var(--fashion-muted);
                font-size: .78rem;
            }}
            .garment-card-type {{ margin-top: 7px; font-weight: 650; }}
            .garment-card-colour {{ margin-top: 3px; }}

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker) {{
                padding: .35rem .35rem .5rem;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 22%, #ece7e1) !important;
                border-radius: 28px 28px 14px 28px !important;
                background: linear-gradient(145deg, rgba(255,255,255,.92), color-mix(in srgb, var(--fashion-cream) 72%, var(--fashion-soft))) !important;
                box-shadow: 0 18px 46px color-mix(in srgb, var(--fashion-primary) 12%, transparent) !important;
            }}
            .recommendation-panel-marker {{
                width: 2.6rem;
                height: .34rem;
                margin: .1rem 0 .15rem;
                border-radius: 999px;
                background: linear-gradient(90deg, var(--fashion-primary), var(--fashion-blush));
            }}
            .recommendation-panel-marker.alternative {{
                background: linear-gradient(90deg, var(--fashion-blush), var(--fashion-secondary));
            }}
            .match-pill {{
                display: inline-flex;
                align-items: center;
                gap: .35rem;
                margin: -.1rem 0 .7rem;
                padding: .3rem .65rem;
                border-radius: 999px;
                background: color-mix(in srgb, var(--fashion-soft) 72%, white);
                color: var(--fashion-strong);
                font-size: .7rem;
                font-weight: 680;
            }}
            .match-pill::before {{
                content: "♥";
                color: color-mix(in srgb, var(--fashion-strong) 72%, #cf7b82);
                font-size: .66rem;
            }}
            div[data-testid="stSelectbox"] > div > div {{
                border-radius: 14px !important;
            }}
            div[data-testid="stFileUploaderDropzone"] {{
                border-radius: 24px 24px 12px 24px !important;
                background: color-mix(in srgb, var(--fashion-cream) 70%, white) !important;
            }}
            .empty-result {{
                border-radius: 28px 28px 14px 28px;
                background: linear-gradient(145deg, rgba(255,255,255,.72), color-mix(in srgb, var(--fashion-cream) 68%, var(--fashion-soft))) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_analysis():
    """Clear selection and analysis data."""

    st.session_state.analysis_result = None
    st.session_state.selected_image = None
    st.session_state.selection_ready = False


def clear_uploaded_image():
    """Remove the uploaded image and reset the workflow."""

    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name = None
    st.session_state.uploader_version += 1
    st.session_state.canvas_version += 1

    reset_analysis()


def select_mode(mode):
    """Enter product or lifestyle mode."""

    st.session_state.image_mode = mode

    clear_uploaded_image()


def return_to_mode_selection():
    """Return to the opening page."""

    st.session_state.image_mode = None

    clear_uploaded_image()


def create_fixed_frame(
    image,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
):
    """
    Fit an image into a fixed-size frame without distortion.

    Returns:
        tuple: framed image and coordinate conversion information.
    """

    image = image.convert("RGB")

    original_width, original_height = image.size

    scale = min(
        frame_width / original_width,
        frame_height / original_height,
    )

    displayed_width = max(
        1,
        int(original_width * scale),
    )

    displayed_height = max(
        1,
        int(original_height * scale),
    )

    resized_image = image.resize(
        (
            displayed_width,
            displayed_height,
        )
    )

    offset_x = (
        frame_width - displayed_width
    ) // 2

    offset_y = (
        frame_height - displayed_height
    ) // 2

    framed_image = Image.new(
        "RGB",
        (
            frame_width,
            frame_height,
        ),
        FRAME_BACKGROUND,
    )

    framed_image.paste(
        resized_image,
        (
            offset_x,
            offset_y,
        ),
    )

    frame_information = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "displayed_width": displayed_width,
        "displayed_height": displayed_height,
    }

    return framed_image, frame_information


def crop_from_drawing(
    original_image,
    drawing_object,
    frame_information,
):
    """
    Convert a freehand drawing boundary back to original-image
    coordinates and return the selected crop.
    """

    left = float(
        drawing_object.get("left", 0)
    )

    top = float(
        drawing_object.get("top", 0)
    )

    width = float(
        drawing_object.get("width", 0)
    )

    height = float(
        drawing_object.get("height", 0)
    )

    drawing_scale_x = float(
        drawing_object.get("scaleX", 1)
    )

    drawing_scale_y = float(
        drawing_object.get("scaleY", 1)
    )

    width *= drawing_scale_x
    height *= drawing_scale_y

    padding = 8

    drawing_left = left - padding
    drawing_top = top - padding
    drawing_right = left + width + padding
    drawing_bottom = top + height + padding

    image_left = frame_information["offset_x"]
    image_top = frame_information["offset_y"]

    image_right = (
        image_left
        + frame_information["displayed_width"]
    )

    image_bottom = (
        image_top
        + frame_information["displayed_height"]
    )

    # Keep the selection inside the displayed image rather than
    # including the frame's empty margins.
    drawing_left = max(
        drawing_left,
        image_left,
    )

    drawing_top = max(
        drawing_top,
        image_top,
    )

    drawing_right = min(
        drawing_right,
        image_right,
    )

    drawing_bottom = min(
        drawing_bottom,
        image_bottom,
    )

    if (
        drawing_right <= drawing_left
        or drawing_bottom <= drawing_top
    ):
        return None

    display_scale = frame_information["scale"]

    original_left = int(
        (drawing_left - image_left)
        / display_scale
    )

    original_top = int(
        (drawing_top - image_top)
        / display_scale
    )

    original_right = int(
        (drawing_right - image_left)
        / display_scale
    )

    original_bottom = int(
        (drawing_bottom - image_top)
        / display_scale
    )

    original_width, original_height = (
        original_image.size
    )

    original_left = max(
        0,
        min(original_left, original_width),
    )

    original_top = max(
        0,
        min(original_top, original_height),
    )

    original_right = max(
        0,
        min(original_right, original_width),
    )

    original_bottom = max(
        0,
        min(original_bottom, original_height),
    )

    if (
        original_right <= original_left
        or original_bottom <= original_top
    ):
        return None

    return original_image.crop(
        (
            original_left,
            original_top,
            original_right,
            original_bottom,
        )
    )


def analyse_clothing(
    selected_image,
    original_image,
    weather_data=None,
):
    """
    Use SigLIP as the primary recognition model.

    The pixel method is retained only as an interpretable
    baseline for debugging and dissertation comparison.
    """

    category_result = predict_category_with_confidence(
        selected_image
    )

    category = category_result["category"]
    recommendation_category = category_result.get(
        "recommendation_category",
        category,
    )

    style_result = predict_style(
        selected_image
    )

    pixel_colour_result = predict_colour_details(
        selected_image,
        reference_image=original_image,
    )

    semantic_colour_result = predict_semantic_colour(
        selected_image
    )

    input_embedding = extract_image_embedding(selected_image)

    pixel_colour = pixel_colour_result["primary_colour"]
    semantic_colour = semantic_colour_result["colour"]

    pixel_distribution = pixel_colour_result.get(
        "colour_distribution",
        {},
    )

    ranked_pixel_colours = sorted(
        pixel_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    colour_palette = [semantic_colour]

    # Black/Grey, White/Grey and White/Beige commonly describe shadows or
    # illumination on one neutral garment rather than intentional multicolour
    # design. Keep the palette conservative for these combinations.
    neutral_shading_pairs = [
        {"Black", "Grey"},
        {"White", "Grey"},
        {"White", "Beige"},
    ]

    # A second colour is retained only when the pixel method finds two
    # substantial regions and SigLIP agrees with one of those colours.
    if len(ranked_pixel_colours) >= 2:
        first_colour, first_share = ranked_pixel_colours[0]
        second_colour, second_share = ranked_pixel_colours[1]
        dominant_pair = {first_colour, second_colour}

        if (
            first_share >= 0.25
            and second_share >= 0.25
            and semantic_colour in dominant_pair
            and dominant_pair not in neutral_shading_pairs
        ):
            other_colour = (
                second_colour
                if semantic_colour == first_colour
                else first_colour
            )

            if other_colour != semantic_colour:
                colour_palette.append(other_colour)

    secondary_colour = (
        colour_palette[1]
        if len(colour_palette) > 1
        else None
    )

    models_agree = (
        pixel_colour == semantic_colour
    )

    siglip_margin = semantic_colour_result.get(
        "margin",
        0.0,
    )

    siglip_scores = sorted(
        (
            float(score)
            for score in semantic_colour_result.get(
                "score_distribution",
                {},
            ).values()
        ),
        reverse=True,
    )

    second_siglip_score = (
        siglip_scores[1]
        if len(siglip_scores) > 1
        else 0.0
    )

    relative_siglip_margin = (
        siglip_margin / siglip_scores[0]
        if siglip_scores and siglip_scores[0] > 0
        else 0.0
    )

    # The final product never interrupts the user with recognition choices.
    # SigLIP supplies the semantic colour; the pixel result remains available
    # only as an interpretable baseline in developer mode.
    final_colour = semantic_colour

    recognised_styles = [
        item["style"]
        for item in style_result.get("styles", [])
    ]

    recommendation = recommend_outfit(
        recommendation_category,
        final_colour,
        styles=recognised_styles,
        weather=weather_data,
        input_embedding=input_embedding,
    )

    if models_agree:
        confirmation_source = "model_agreement"
    elif relative_siglip_margin >= 0.35:
        confirmation_source = "siglip_clear_lead"
    else:
        confirmation_source = "siglip_automatic_fallback"

    return {
        "category": category,
        "recommendation_category": recommendation_category,
        "category_result": category_result,
        "style_result": style_result,
        "pixel_colour": pixel_colour,
        "pixel_colour_result": pixel_colour_result,
        "semantic_colour": semantic_colour,
        "semantic_colour_result": semantic_colour_result,
        "colour_palette": colour_palette,
        "secondary_colour": secondary_colour,
        "is_multicolour": len(colour_palette) > 1,
        "models_agree": models_agree,
        "colour": final_colour,
        "recommendation": recommendation,
        "confirmation_source": confirmation_source,
        "siglip_margin": siglip_margin,
        "siglip_relative_margin": relative_siglip_margin,
        "siglip_second_score": second_siglip_score,
        "weather": weather_data,
        "input_embedding": input_embedding,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_cached_weather(city):
    """Cache a city's forecast for fifteen minutes."""

    return get_city_weather(city)


def ensure_current_weather():
    """Load weather on entry and retain it for analysis and display."""

    try:
        weather_data = get_cached_weather(
            st.session_state.city.strip()
        )
        st.session_state.weather_data = weather_data
        st.session_state.weather_error = None
    except WeatherServiceError as error:
        weather_data = None
        st.session_state.weather_data = None
        st.session_state.weather_error = str(error)
    return weather_data


def weather_dressing_advice(weather):
    """Turn current conditions into one short wardrobe reminder."""

    feels_like = float(weather.get("feels_like", 20.0))
    rain_probability = float(weather.get("rain_probability", 0.0))
    condition = weather.get("condition", "Unknown")
    if rain_probability >= 50 or condition in {"Rain", "Thunderstorm"}:
        return "Prioritise a waterproof outer layer and water-resistant shoes."
    if condition == "Snow" or feels_like <= 5:
        return "Choose thermal layers, an insulated coat and warm footwear."
    if feels_like <= 12:
        return "A warm jacket and a comfortable mid-layer will work well today."
    if feels_like >= 28:
        return "Keep it breathable and light; an outer layer is probably unnecessary."
    if feels_like >= 24:
        return "Choose breathable fabrics and keep any outer layer lightweight."
    return "Comfortable layers are ideal; carry a light jacket for temperature changes."


def render_weather_window(weather=None, error=None):
    """Render a lightweight, ambient weather window for the selected city."""

    condition = str((weather or {}).get("condition", "Clear"))
    scene_classes = {
        "Clear": "clear",
        "Cloudy": "cloudy",
        "Fog": "fog",
        "Rain": "rain",
        "Snow": "snow",
        "Thunderstorm": "thunderstorm",
    }
    scene = scene_classes.get(condition, "clear")
    city = html.escape(
        str((weather or {}).get("city", st.session_state.city))
    )
    country = html.escape(str((weather or {}).get("country", "")))
    location = f"{city}, {country}" if country else city

    if weather:
        scene_notes = {
            "Clear": "Sunshine is keeping you company",
            "Cloudy": "Soft clouds are drifting by",
            "Fog": "The clouds have come down to rest",
            "Rain": "Little raindrops are tapping the glass",
            "Snow": "Snowflakes have come to say hello",
            "Thunderstorm": "The sky is feeling a little dramatic",
        }
        note = scene_notes.get(condition, "The weather is keeping you company")
        temperature = f'{float(weather.get("temperature", weather.get("feels_like", 20))):.0f}°'
        accessibility = f"Calm animated {condition.lower()} weather outside {location}"
    else:
        note = (
            "The forecast is resting — your room stays warm"
            if error
            else "Set your city to wake the window"
        )
        temperature = "♡"
        accessibility = "A calm warm window waiting for a city forecast"

    st.markdown(
        f"""
        <div class="weather-window-scene weather-{scene}" role="img"
             aria-label="{html.escape(accessibility)}">
            <div class="weather-scene-label">Outside · {location}</div>
            <div class="weather-illustration" aria-hidden="true">
                <div class="weather-orb"></div>
                <div class="weather-cloud-shadow"></div>
                <div class="weather-cloud-form"></div>
                <div class="weather-face">
                    <i class="eye left"></i><i class="eye right"></i>
                    <i class="mouth"></i>
                    <i class="cheek left"></i><i class="cheek right"></i>
                </div>
                <div class="weather-sparkles"><i></i><i></i></div>
                <div class="weather-precipitation">
                    <i style="--drop: 0"></i><i style="--drop: 1"></i>
                    <i style="--drop: 2"></i><i style="--drop: 3"></i>
                    <i style="--drop: 4"></i><i style="--drop: 5"></i>
                    <i style="--drop: 6"></i>
                </div>
                <div class="weather-mist"><i></i><i></i><i></i></div>
                <div class="weather-bolt"></div>
            </div>
            <div class="weather-glass-light"></div>
            <div class="weather-window-info">
                <div class="weather-window-copy">
                    <strong>{location}</strong><span>{note}</span>
                </div>
                <div class="weather-window-temperature">{temperature}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_weather(weather, error=None):
    """Render the persistent left-column weather and dressing card."""

    if weather is None:
        message = html.escape(error or "Live weather is temporarily unavailable.")
        st.markdown(
            f"""
            <div class="weather-card">
                <div class="weather-head">
                    <div>
                        <div class="weather-location">Weather unavailable</div>
                        <div class="weather-condition">{message}</div>
                    </div>
                    <div class="weather-icon">◌</div>
                </div>
                <div class="weather-advice"><strong>Dressing cue</strong><br>
                Use colour and style recommendations until the forecast returns.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    condition = str(weather.get("condition", "Unknown"))
    weather_icons = {
        "Clear": "☀️",
        "Cloudy": "⛅",
        "Fog": "🌫️",
        "Rain": "🌧️",
        "Snow": "❄️",
        "Thunderstorm": "⛈️",
    }
    icon = weather_icons.get(condition, "🌤️")
    city = html.escape(str(weather.get("city", st.session_state.city)))
    country = html.escape(str(weather.get("country", "")))
    location = f"{city}, {country}" if country else city
    advice = html.escape(weather_dressing_advice(weather))
    st.markdown(
        f"""
        <div class="weather-card">
            <div class="weather-head">
                <div>
                    <div class="weather-location">{location}</div>
                    <div class="weather-condition">{html.escape(condition)} today</div>
                </div>
                <div class="weather-icon">{icon}</div>
            </div>
            <div class="weather-stats">
                <div class="weather-stat">
                    <div class="weather-stat-label">Feels like</div>
                    <div class="weather-stat-value">{weather['feels_like']:.0f}°C</div>
                </div>
                <div class="weather-stat">
                    <div class="weather-stat-label">Rain</div>
                    <div class="weather-stat-value">{weather['rain_probability']:.0f}%</div>
                </div>
                <div class="weather-stat">
                    <div class="weather-stat-label">Wind</div>
                    <div class="weather-stat-value">{weather['wind_speed']:.0f} km/h</div>
                </div>
            </div>
            <div class="weather-advice"><strong>Dressing cue</strong><br>{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyse_with_current_weather(selected_image, original_image):
    """Run recognition with weather, falling back safely if unavailable."""

    weather_data = ensure_current_weather()

    return analyse_clothing(
        selected_image,
        original_image,
        weather_data=weather_data,
    )


def recommendation_inputs(result):
    """Preserve model inputs when a user changes style, category or colour."""

    return {
        "styles": [
            item["style"]
            for item in result.get("style_result", {}).get("styles", [])
        ],
        "weather": result.get("weather"),
        "input_embedding": result.get("input_embedding"),
    }

def prepare_score_rows(score_distribution, limit=6):
    """Convert a score dictionary into compact sorted table rows."""

    if not score_distribution:
        return []

    sorted_scores = sorted(
        score_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    highest_score = float(sorted_scores[0][1]) if sorted_scores else 0.0

    return [
        {
            "Label": label,
            "Raw score": f"{float(score):.3%}",
            "Relative": (
                f"{float(score) / highest_score:.0%}"
                if highest_score > 0
                else "0%"
            ),
        }
        for label, score in sorted_scores[:limit]
    ]


@st.dialog(
    "Full Debug Report",
    width="large",
)
def show_debug_report(
    result,
    original_image,
    selected_image,
):
    """
    Display recognition inputs and intermediate results
    in one compact debugging window.
    """

    pixel_result = result["pixel_colour_result"]
    semantic_result = result["semantic_colour_result"]
    category_result = result["category_result"]
    style_result = result["style_result"]

    selected_image = (
        selected_image
        if selected_image is not None
        else original_image
    )

    original_preview, _ = create_fixed_frame(
        original_image,
        frame_width=340,
        frame_height=230,
    )

    selected_preview, _ = create_fixed_frame(
        selected_image,
        frame_width=340,
        frame_height=230,
    )

    image_column, summary_column = st.columns(
        [1.05, 1.25],
        gap="medium",
    )

    with image_column:
        original_column, selected_column = st.columns(2)

        with original_column:
            st.markdown("#### Original image")
            st.image(
                original_preview,
                use_container_width=True,
            )
            st.caption(
                f"Original size: "
                f"{original_image.width} × "
                f"{original_image.height}"
            )

        with selected_column:
            st.markdown("#### Analysed region")
            st.image(
                selected_preview,
                use_container_width=True,
            )
            st.caption(
                f"Crop size: "
                f"{selected_image.width} × "
                f"{selected_image.height}"
            )

        original_area = (
            original_image.width
            * original_image.height
        )

        selected_area = (
            selected_image.width
            * selected_image.height
        )

        crop_ratio = (
            selected_area / original_area
            if original_area
            else 0
        )

        st.caption(
            f"Selected area: {crop_ratio:.1%} "
            "of the original image"
        )

    with summary_column:
        st.markdown("#### Recognition summary")

        summary_one, summary_two, summary_three = (
            st.columns(3)
        )

        with summary_one:
            st.metric(
                "Category",
                result["category"],
            )

            if category_result.get("fine_category_confident", False):
                st.caption(
                    "Parent: "
                    f"{category_result.get('parent_category', 'Unknown')}"
                )
            elif result["category"] != "Unknown":
                likely_categories = category_result.get(
                    "likely_fine_categories",
                    [],
                )
                st.caption(
                    "Possible fine categories: "
                    + " / ".join(likely_categories[:2])
                )
            else:
                st.caption(
                    "Suggested category: "
                    f"{category_result.get('predicted_fine_category', 'Unknown')}"
                )

        with summary_two:
            st.metric(
                "Pixel colour",
                result["pixel_colour"],
            )

        with summary_three:
            st.metric(
                "SigLIP colour",
                result["semantic_colour"],
            )

        recognised_styles = ", ".join(
            style["style"]
            for style in style_result.get("styles", [])
        )

        st.caption(
            "Recognised style cues: "
            f"{recognised_styles or 'Unknown'}"
        )

        st.caption(
            "Colour palette: "
            + " + ".join(result.get("colour_palette", []))
        )

        summary_four, summary_five, summary_six = (
            st.columns(3)
        )

        with summary_four:
            st.metric(
                "Final colour",
                result["colour"] or "Unconfirmed",
            )

        with summary_five:
            st.metric(
                "Models agree",
                "Yes"
                if result["models_agree"]
                else "No",
            )

        with summary_six:
            st.metric(
                "Final source",
                result["confirmation_source"],
            )

        score_one, score_two, score_three = st.columns(3)

        with score_one:
            st.metric(
                "Category SigLIP score",
                (
                    f"{category_result.get('confidence', 0):.0%}"
                ),
            )

        with score_two:
            st.metric(
                "Pixel certainty",
                f"{pixel_result.get('confidence', 0):.0%}",
            )

        with score_three:
            st.metric(
                "SigLIP colour score",
                (
                    f"{semantic_result.get('confidence', 0):.0%}"
                ),
            )

        recommendation = result.get("recommendation")

        st.markdown("#### Recommendation state")

        if recommendation is None:
            if result["category"] == "Unknown":
                st.warning(
                    "Recommendation is waiting for category "
                    "confirmation."
                )
            else:
                st.warning(
                    "Recommendation is waiting for colour "
                    "confirmation."
                )
        else:
            recommended_items = " | ".join(
                recommendation.get("items", [])
            )

            st.success(recommended_items)

            st.caption(
                "Method: "
                f"{recommendation.get('method', 'Unknown')}"
            )

    st.divider()

    (
        category_column,
        style_column,
        colour_column,
        processing_column,
    ) = (
        st.columns(
            [1, 1, 1, 1],
            gap="medium",
        )
    )

    with category_column:
        st.markdown("#### Category SigLIP scores")

        category_rows = prepare_score_rows(
            category_result.get(
                "score_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            category_rows,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with style_column:
        st.markdown("#### SigLIP style scores")

        style_rows = prepare_score_rows(
            style_result.get(
                "score_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            style_rows,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with colour_column:
        st.markdown("#### SigLIP colour scores")

        semantic_rows = prepare_score_rows(
            semantic_result.get(
                "score_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            semantic_rows,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with processing_column:
        st.markdown("#### Pixel colour distribution")

        pixel_rows = prepare_score_rows(
            pixel_result.get(
                "colour_distribution",
                {},
            ),
            limit=7,
        )

        st.dataframe(
            pixel_rows,
            use_container_width=True,
            hide_index=True,
            height=160,
        )

        gains = pixel_result.get(
            "white_balance_gains",
            {},
        )

        gain_rows = [
            {
                "Channel": channel.title(),
                "Gain": round(float(value), 3),
            }
            for channel, value in gains.items()
        ]

        st.markdown("#### White-balance gains")

        st.dataframe(
            gain_rows,
            use_container_width=True,
            hide_index=True,
            height=120,
        )

def render_results(
    result,
    original_image,
    selected_image,
):
    """
    Display category, two colour methods and recommendation results.
    """

    all_colours = [
        "Black",
        "White",
        "Grey",
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Blue",
        "Purple",
        "Pink",
        "Brown",
        "Beige",
    ]

    category = result["category"]
    category_result = result["category_result"]
    recommendation_category = result.get(
        "recommendation_category",
        category,
    )
    style_result = result["style_result"]
    pixel_colour = result["pixel_colour"]
    semantic_colour = result["semantic_colour"]

    pixel_result = result["pixel_colour_result"]
    semantic_result = result["semantic_colour_result"]

    with st.container(border=True):
        st.subheader("Recognition Result")

        category_column, style_column, colour_column = st.columns(3)

        with category_column:
            st.metric(
                "Category",
                category,
            )

            if category_result.get(
                "fine_category_confident",
                False,
            ):
                st.caption(
                    "Parent: "
                    f"{category_result.get('parent_category', 'Unknown')}"
                )
            elif category != "Unknown":
                likely_categories = category_result.get(
                    "likely_fine_categories",
                    [],
                )

                st.caption(
                    "Possible: "
                    + " / ".join(likely_categories[:2])
                )

                st.caption(
                    "Fine category uncertain; the broad category "
                    "is used for recommendation."
                )

        with style_column:
            st.metric(
                "Primary style",
                style_result["primary_style"],
            )

            secondary_styles = [
                item["style"]
                for item in style_result.get("styles", [])[1:]
            ]

            if secondary_styles:
                st.caption(
                    "Also: "
                    + ", ".join(secondary_styles)
                )

        with colour_column:
            final_colour = (
                result["colour"]
                if result["colour"] is not None
                else "Confirm below"
            )

            st.metric(
                "Final colour",
                final_colour,
            )

            colour_palette = result.get("colour_palette", [])

            if len(colour_palette) > 1:
                st.caption(
                    "Palette: "
                    + " + ".join(colour_palette)
                )

        st.markdown("#### Colour Comparison")

        pixel_column, semantic_column = st.columns(2)

        with pixel_column:
            st.metric(
                "Pixel method",
                pixel_colour,
                help=(
                    "White balance, Lab colour space "
                    "and K-means clustering."
                ),
            )

            st.caption(
                "Certainty score: "
                f"{pixel_result['confidence']:.0%}"
            )

        with semantic_column:
            st.metric(
                "SigLIP semantic method",
                semantic_colour,
                help=(
                    "Zero-shot visual-language "
                    "classification."
                ),
            )

            st.caption(
                "SigLIP score: "
                f"{semantic_result['confidence']:.0%}"
            )

        if st.button(
            "Open Full Debug Report",
            use_container_width=True,
            key="open_debug_report_button",
        ):
            show_debug_report(
                result,
                original_image,
                selected_image,
            )

        needs_category_confirmation = (
            category == "Unknown"
        )

        if needs_category_confirmation:
            predicted_fine_category = result[
                "category_result"
            ].get(
                "predicted_fine_category",
                "Unknown",
            )

            st.warning(
                "The garment does not strongly match the current "
                "category set. Please confirm its category."
            )

            suggested_index = (
                CATEGORY_CHOICES.index(predicted_fine_category)
                if predicted_fine_category in CATEGORY_CHOICES
                else 0
            )

            confirmed_category = st.selectbox(
                "Confirm clothing category",
                CATEGORY_CHOICES,
                index=suggested_index,
                key="unknown_category_choice",
            )

            if st.button(
                "Use Confirmed Category",
                type="primary",
                use_container_width=True,
                key="confirm_category_button",
            ):
                result["category"] = confirmed_category
                result["recommendation_category"] = (
                    CATEGORY_RECOMMENDATION_ALIASES.get(
                        confirmed_category,
                        confirmed_category,
                    )
                )
                result["category_result"][
                    "category_out_of_scope"
                ] = False
                result["category_result"][
                    "parent_category"
                ] = CATEGORY_PARENTS[confirmed_category]
                result["category_result"][
                    "fine_category_confident"
                ] = True
                result["category_confirmation_source"] = (
                    "user_confirmation"
                )

                if result["colour"] is not None:
                    result["recommendation"] = recommend_outfit(
                        result["recommendation_category"],
                        result["colour"],
                        **recommendation_inputs(result),
                    )

                st.session_state.analysis_result = result
                st.rerun()

            return

        needs_confirmation = (
            result["colour"] is None
        )

        if needs_confirmation:
            st.warning(
                "The colour result is uncertain. "
                "Please confirm the clothing colour."
            )

            confirmation_options = []

            for colour in [
                semantic_colour,
                pixel_colour,
                *all_colours,
            ]:
                if colour not in confirmation_options:
                    confirmation_options.append(colour)

            selected_colour = st.selectbox(
                "Confirm clothing colour",
                confirmation_options,
                key="colour_disagreement_choice",
            )

            if st.button(
                "Use Confirmed Colour",
                type="primary",
                use_container_width=True,
                key="confirm_colour_button",
            ):
                result["colour"] = selected_colour
                result["colour_palette"] = [selected_colour]
                result["secondary_colour"] = None
                result["is_multicolour"] = False

                result["recommendation"] = recommend_outfit(
                    recommendation_category,
                    selected_colour,
                    **recommendation_inputs(result),
                )

                result["confirmation_source"] = (
                    "user_confirmation"
                )

                st.session_state.analysis_result = result

                st.rerun()

            # Do not show a recommendation until colour is confirmed.
            return

        recommendation = result["recommendation"]

        st.markdown("#### Recommended Outfit")

        st.caption(
            f"Method: {recommendation['method']}"
        )

        recommendation_columns = st.columns(
            len(recommendation["items"])
        )

        for column, item in zip(
            recommendation_columns,
            recommendation["items"],
        ):
            with column:
                st.markdown(
                    (
                        '<div class="result-item">'
                        f"{item}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        with st.expander("Why this works"):
            st.write(
                recommendation["explanation"]
            )

        with st.expander("Correct category manually"):
            predicted_fine_category = category_result.get(
                "predicted_fine_category",
                category,
            )

            category_index = (
                CATEGORY_CHOICES.index(predicted_fine_category)
                if predicted_fine_category in CATEGORY_CHOICES
                else 0
            )

            corrected_category = st.selectbox(
                "Correct category",
                CATEGORY_CHOICES,
                index=category_index,
                key="manual_category_override",
            )

            if st.button(
                "Apply Category Correction",
                use_container_width=True,
                key="apply_category_correction",
            ):
                corrected_recommendation_category = (
                    CATEGORY_RECOMMENDATION_ALIASES.get(
                        corrected_category,
                        corrected_category,
                    )
                )

                result["category"] = corrected_category
                result["recommendation_category"] = (
                    corrected_recommendation_category
                )
                result["category_result"][
                    "predicted_fine_category"
                ] = corrected_category
                result["category_result"][
                    "parent_category"
                ] = CATEGORY_PARENTS[corrected_category]
                result["category_result"][
                    "fine_category_confident"
                ] = True
                result["category_confirmation_source"] = (
                    "manual_correction"
                )
                result["recommendation"] = recommend_outfit(
                    corrected_recommendation_category,
                    result["colour"],
                    **recommendation_inputs(result),
                )

                st.session_state.analysis_result = result
                st.rerun()

        # Manual correction remains available even when models agree.
        with st.expander("Correct colour manually"):
            current_colour = result["colour"]

            current_index = all_colours.index(
                current_colour
            )

            corrected_colour = st.selectbox(
                "Correct colour",
                all_colours,
                index=current_index,
                key="manual_colour_override",
            )

            if st.button(
                "Apply Correction",
                use_container_width=True,
                key="apply_colour_correction",
            ):
                result["colour"] = corrected_colour
                result["colour_palette"] = [corrected_colour]
                result["secondary_colour"] = None
                result["is_multicolour"] = False

                result["recommendation"] = recommend_outfit(
                    recommendation_category,
                    corrected_colour,
                    **recommendation_inputs(result),
                )

                result["confirmation_source"] = (
                    "manual_correction"
                )

                st.session_state.analysis_result = result

                st.rerun()

        st.markdown("#### Current Test Data")

        agreement_column, source_column = st.columns(2)

        with agreement_column:
            st.metric(
                "Models agree",
                "Yes" if result["models_agree"] else "No",
            )

        with source_column:
            st.metric(
                "Final source",
                result["confirmation_source"],
            )

        with st.expander("Detailed test data"):
            st.write("Pixel colour distribution")

            st.json(
                pixel_result["colour_distribution"]
            )

            st.write("SigLIP colour distribution")

            st.json(
                semantic_result["score_distribution"]
            )

            st.write("White-balance gains")

            st.json(
                pixel_result["white_balance_gains"]
            )

            st.write("Category SigLIP distribution")

            st.json(
                result["category_result"][
                    "score_distribution"
                ]
            )

            st.write("Style SigLIP distribution")

            st.json(
                style_result["score_distribution"]
            )


def garment_icon_svg(slot, colour, item_type=""):
    """Return a compact garment icon coloured to match the recommendation."""

    fill = ICON_COLOURS.get(colour, "#e8c95e")
    stroke = "#34363d"

    paths = {
        "inner_top": (
            '<path d="M27 18 38 11h20l11 7 13 18-12 8-8-10v48H34V34l-8 10-12-8z"/>'
        ),
        "tank_top": (
            '<path d="M37 11h8c0 8 6 11 11 0h7l7 17-9 5v49H35V33l-8-5z"/>'
            '<path d="M43 12c0 9 10 9 12 0" fill="none"/>'
        ),
        "shorts": (
            '<path d="M31 15h34l3 48-17-1-3-20-3 20-17 1z"/>'
            '<path d="M31 28h34" fill="none"/>'
        ),
        "skirt": (
            '<path d="M36 14h24l13 66H23z"/>'
            '<path d="M34 27h28" fill="none"/>'
        ),
        "dress": (
            '<path d="M39 10h18l5 18 18 53H16l18-53z"/>'
            '<path d="M36 30h24" fill="none"/>'
        ),
        "no_layer": (
            '<circle cx="48" cy="48" r="18"/>'
            '<path d="M48 10v12M48 74v12M10 48h12M74 48h12M21 21l9 9M66 66l9 9M75 21l-9 9M30 66l-9 9" fill="none"/>'
        ),
        "outer_top": (
            '<path d="M28 17 40 10h16l12 7 14 20-12 7-8-12v50H34V32l-8 12-12-7z"/>'
            '<path d="M48 11v71M39 34h18" fill="none"/>'
        ),
        "bottom": (
            '<path d="M34 12h28l5 70H52l-4-44-4 44H29z"/>'
            '<path d="M34 25h28" fill="none"/>'
        ),
        "shoes": (
            '<path d="M17 53c13 0 19-8 25-18l12 15c5 6 12 9 24 10v14H17z"/>'
            '<path d="M53 68h25" fill="none"/>'
        ),
        "trainers": (
            '<path d="M15 54c14 0 21-7 28-20l12 15c6 7 13 10 27 12v13H15z"/>'
            '<path d="m43 44 15 13M37 51l9 7M16 67h66" fill="none"/>'
        ),
        "leather_shoes": (
            '<path d="M17 52c15 1 24-5 31-16l9 14c6 7 12 8 23 10v14H17z"/>'
            '<path d="M45 48h15M17 68h63" fill="none"/>'
        ),
        "boots": (
            '<path d="M30 13h28v42c5 4 12 6 23 7v13H26V55h4z"/>'
            '<path d="M31 52h28M26 68h55" fill="none"/>'
        ),
        "sandals": (
            '<path d="M18 63c15 0 24-5 33-17 8 8 17 12 29 15v13H18z"/>'
            '<path d="M39 53 53 68M54 50l12 16M18 68h62" fill="none"/>'
        ),
    }

    item_name = item_type.casefold()
    if "too hot" in item_name or "no outer" in item_name:
        icon_name = "no_layer"
    elif any(name in item_name for name in ("tank", "vest", "cami")):
        icon_name = "tank_top"
    elif "short" in item_name:
        icon_name = "shorts"
    elif "skirt" in item_name:
        icon_name = "skirt"
    elif "dress" in item_name and slot != "shoes":
        icon_name = "dress"
    elif "sandal" in item_name:
        icon_name = "sandals"
    elif "boot" in item_name:
        icon_name = "boots"
    elif any(
        name in item_name
        for name in ("loafer", "leather", "dress shoe")
    ):
        icon_name = "leather_shoes"
    elif any(
        name in item_name
        for name in ("trainer", "sneaker", "running", "trail")
    ):
        icon_name = "trainers"
    else:
        icon_name = slot

    path = paths.get(icon_name, paths["inner_top"])
    return (
        '<svg viewBox="0 0 96 96" width="76" height="76" '
        'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        f'<g fill="{fill}" stroke="{stroke}" stroke-width="3" '
        f'stroke-linejoin="round">{path}</g></svg>'
    )


def render_outfit_cards(outfit):
    """Render one three-item outfit with coloured category icons."""

    columns = st.columns(3, gap="medium")

    for column, item in zip(columns, outfit["items"]):
        with column:
            colour_line = (
                f'<div class="garment-card-colour">{html.escape(item["colour"])}</div>'
                if item.get("colour")
                else ""
            )
            st.markdown(
                (
                    '<div class="garment-card">'
                    '<div class="garment-icon-shell">'
                    f'{garment_icon_svg(item["slot"], item["colour"], item["type"])}'
                    '</div>'
                    f'<div class="garment-card-slot">{html.escape(item["slot_label"])}</div>'
                    f'<div class="garment-card-type">{html.escape(item["type"])}</div>'
                    f'{colour_line}</div>'
                ),
                unsafe_allow_html=True,
            )


def render_recommendations_only(result):
    """Render recognised styles and two complete recommendations."""

    recommendation = result["recommendation"]

    # Rebuild results created before the four-slot recommender was loaded.
    if "primary" not in recommendation:
        recognised_styles = [
            item["style"]
            for item in result["style_result"].get("styles", [])
        ]
        recommendation = recommend_outfit(
            result["recommendation_category"],
            result["colour"],
            styles=recognised_styles,
            weather=result.get("weather"),
            input_embedding=result.get("input_embedding"),
        )
        result["recommendation"] = recommendation

    weather = result.get("weather")

    recognised_styles = [
        item["style"]
        for item in result["style_result"].get("styles", [])
    ] or ["Casual"]

    with st.container(border=True):
        st.markdown(
            '<div class="recommendation-panel-marker"></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Style")
        st.caption(
            "Recognised: " + ", ".join(recognised_styles)
        )

        style_column, button_column = st.columns([1.35, 0.65])
        with style_column:
            selected_style = st.selectbox(
                "Choose a style",
                STYLE_OPTIONS,
                index=STYLE_OPTIONS.index(recognised_styles[0])
                if recognised_styles[0] in STYLE_OPTIONS
                else 0,
                key="recommendation_style_choice",
            )
        with button_column:
            st.write("")
            if st.button(
                "Recommend Again",
                use_container_width=True,
                type="primary",
            ):
                result["recommendation"] = recommend_outfit(
                    result["recommendation_category"],
                    result["colour"],
                    styles=recognised_styles,
                    weather=weather,
                    selected_style=selected_style,
                    input_embedding=result.get("input_embedding"),
                )
                st.session_state.analysis_result = result
                st.rerun()

    recommendation = result["recommendation"]

    with st.container(border=True):
        st.markdown(
            '<div class="recommendation-panel-marker"></div>',
            unsafe_allow_html=True,
        )
        st.subheader(
            f'Primary Outfit · {recommendation["primary"]["style"]}'
        )
        if "model_score" in recommendation["primary"]:
            st.markdown(
                '<div class="match-pill">Outfit match&nbsp; '
                f'{recommendation["primary"]["model_score"]:.1f}/100</div>',
                unsafe_allow_html=True,
            )
        render_outfit_cards(recommendation["primary"])

    with st.container(border=True):
        st.markdown(
            '<div class="recommendation-panel-marker alternative"></div>',
            unsafe_allow_html=True,
        )
        st.subheader(
            f'Alternative Outfit · {recommendation["alternative"]["style"]}'
        )
        if "model_score" in recommendation["alternative"]:
            st.markdown(
                '<div class="match-pill">Outfit match&nbsp; '
                f'{recommendation["alternative"]["model_score"]:.1f}/100</div>',
                unsafe_allow_html=True,
            )
        render_outfit_cards(recommendation["alternative"])


initialise_state()
render_theme_css()

if st.session_state.image_mode is not None:
    st.markdown(
        '<div class="app-wordmark">AI Outfit Recommendation</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# Opening mode-selection page
# =========================================================

if st.session_state.image_mode is None:
    st.markdown(
        """
        <style>
            .stButton > button {
                min-height: 56px;
                padding: .95rem 1.4rem;
                border: 1px solid var(--fashion-strong);
                border-radius: 999px;
                background: var(--fashion-strong);
                color: white;
                box-shadow: 0 13px 30px color-mix(in srgb, var(--fashion-strong) 24%, transparent);
                font-size: .82rem;
                letter-spacing: .12em;
                line-height: 1.2;
                text-align: center;
                white-space: pre-wrap;
                transition:
                    transform 0.2s ease,
                    border-color 0.2s ease,
                    box-shadow 0.2s ease;
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                border-color: var(--fashion-strong);
                color: white !important;
                filter: brightness(.94);
                box-shadow:
                    0 18px 38px
                    color-mix(in srgb, var(--fashion-primary) 22%, transparent);
            }

            div[data-testid="stRadio"] label p {
                font-size: .76rem;
            }

            @media (max-width: 760px) {
                .opening-title {
                    font-size: clamp(2.25rem, 11vw, 3.25rem);
                    line-height: 1;
                }
                .home-wordmark { margin-bottom: 1.2rem; }
                .weather-window-scene { min-height: 520px; margin-top: 1.4rem; }
                .weather-illustration { inset: 3rem .65rem 6rem; }
                .weather-orb { width: 8rem; right: 9%; }
                .weather-cloud-form { left: 10%; width: 66%; }
                .weather-window-temperature { font-size: 1.7rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    home_weather = st.session_state.weather_data
    home_weather_error = st.session_state.weather_error

    copy_column, image_column = st.columns(
        [0.88, 1.12],
        gap="large",
        vertical_alignment="center",
    )

    with copy_column:
        st.markdown(
            """
            <div class="home-wordmark">
                <span class="home-brand">Your Wardrobe</span>
                <span class="home-edition">Daily styling</span>
            </div>
            <div class="opening-kicker">Your daily outfit companion</div>
            <div class="opening-title">Start with one piece.</div>
            <div class="opening-subtitle">
                We’ll style the rest around your mood and the weather outside.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                """
                <div class="city-panel-marker">
                    <div class="city-panel-icon">⌖</div>
                    <div>
                        <div class="city-panel-title">Where are you dressing for?</div>
                        <div class="city-panel-copy">Your forecast sets the layers.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("weather_city_form", border=False):
                city_column, save_city_column = st.columns([0.68, 0.32])
                with city_column:
                    st.text_input(
                        "City",
                        key="city_input",
                        placeholder="Shanghai",
                        label_visibility="collapsed",
                        help="Used for today's temperature, rain and wind.",
                    )
                with save_city_column:
                    city_submitted = st.form_submit_button(
                        "Set location",
                        use_container_width=True,
                    )

                if city_submitted:
                    entered_city = st.session_state.city_input.strip()
                    if entered_city:
                        st.session_state.city = entered_city
                        st.session_state.weather_data = None
                        st.session_state.weather_error = None
                        with st.spinner("Opening your weather window..."):
                            home_weather = ensure_current_weather()
                        home_weather_error = st.session_state.weather_error

        st.radio(
            "Choose your mood",
            list(THEMES),
            horizontal=True,
            key="theme",
        )

        if st.button(
            "START WITH A PIECE  →",
            type="primary",
            use_container_width=True,
            key="style_mode_button",
        ):
            select_mode("lifestyle")
            st.rerun()

        st.markdown(
            '<div class="home-greeting">Your clothes, your mood, your little world.</div>',
            unsafe_allow_html=True,
        )

    with image_column:
        render_weather_window(
            home_weather,
            home_weather_error,
        )

    st.stop()


# =========================================================
# Working page
# =========================================================

mode_title = "What do you want to wear today?"

current_weather = ensure_current_weather()

left_column, right_column = st.columns(
    [0.95, 1.05],
    gap="large",
)


# ---------------------------------------------------------
# Left-column navigation
# ---------------------------------------------------------

with left_column:
    navigation_column, title_column = (
        st.columns(
            [0.40, 0.60],
            vertical_alignment="center",
        )
    )

    with navigation_column:
        if st.button(
            "← Back",
            use_container_width=True,
            key="back_button",
        ):
            return_to_mode_selection()
            st.rerun()

    with title_column:
        st.subheader(mode_title)


# ---------------------------------------------------------
# Upload image only when no image is stored
# ---------------------------------------------------------

if st.session_state.uploaded_bytes is None:
    with left_column:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=(
                "uploader_"
                f"{st.session_state.image_mode}_"
                f"{st.session_state.uploader_version}"
            ),
        )

        if uploaded_file is not None:
            st.session_state.uploaded_bytes = (
                uploaded_file.getvalue()
            )

            st.session_state.uploaded_name = (
                uploaded_file.name
            )

            st.session_state.canvas_version += 1

            reset_analysis()

            st.rerun()

    with right_column:
        st.markdown(
            """
            <div class="empty-result">
                Upload an image to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with left_column:
        render_compact_weather(
            current_weather,
            st.session_state.weather_error,
        )

    st.stop()


# ---------------------------------------------------------
# Load stored original image
# ---------------------------------------------------------

original_image = Image.open(
    io.BytesIO(
        st.session_state.uploaded_bytes
    )
).convert("RGB")


# ---------------------------------------------------------
# Left-column image controls
# ---------------------------------------------------------

with left_column:
    if st.button(
        "↻ Change Image",
        use_container_width=True,
        key="change_image_button",
    ):
        clear_uploaded_image()
        st.rerun()

    if st.session_state.image_mode == "product":
        framed_product, _ = create_fixed_frame(
            original_image
        )

        st.image(
            framed_product,
            width=FRAME_WIDTH,
        )

        st.session_state.selected_image = (
            original_image
        )

        st.session_state.selection_ready = True

        if st.button(
            "Analyse Item",
            type="primary",
            use_container_width=True,
            key="analyse_product_button",
        ):
            with st.spinner("Analysing..."):
                st.session_state.analysis_result = (
                    analyse_with_current_weather(
                        original_image,
                        original_image,
                    )
                )

            st.rerun()

    else:
        # -------------------------------------------------
        # Lifestyle mode: original canvas
        # -------------------------------------------------

        if not st.session_state.selection_ready:
            canvas_frame, frame_information = (
                create_fixed_frame(
                    original_image
                )
            )

            canvas_result = st_canvas(
                background_image=canvas_frame,
                drawing_mode="freedraw",
                stroke_width=6,
                stroke_color=THEMES[st.session_state.theme]["canvas"],
                fill_color=THEMES[st.session_state.theme]["canvas_fill"],
                update_streamlit=True,
                display_toolbar=True,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                key=(
                    "canvas_"
                    f"{st.session_state.uploaded_name}_"
                    f"{st.session_state.canvas_version}"
                ),
            )

            canvas_objects = []

            if canvas_result.json_data is not None:
                canvas_objects = (
                    canvas_result.json_data.get(
                        "objects",
                        [],
                    )
                )

            if canvas_objects:
                latest_drawing = (
                    canvas_objects[-1]
                )

                selected_image = (
                    crop_from_drawing(
                        original_image,
                        latest_drawing,
                        frame_information,
                    )
                )

                if selected_image is not None:
                    st.session_state.selected_image = (
                        selected_image
                    )

                    st.session_state.selection_ready = (
                        True
                    )

                    st.session_state.analysis_result = (
                        None
                    )

                    st.rerun()

                else:
                    st.warning(
                        "Please draw around the "
                        "clothing item itself."
                    )

        # -------------------------------------------------
        # Lifestyle mode: selected item only
        # -------------------------------------------------

        else:
            selected_image = (
                st.session_state.selected_image
            )

            framed_selection, _ = (
                create_fixed_frame(
                    selected_image
                )
            )

            st.image(
                framed_selection,
                width=FRAME_WIDTH,
            )

            redraw_column, analyse_column = (
                st.columns(2)
            )

            with redraw_column:
                if st.button(
                    "Draw Again",
                    use_container_width=True,
                    key="draw_again_button",
                ):
                    st.session_state.selected_image = (
                        None
                    )

                    st.session_state.selection_ready = (
                        False
                    )

                    st.session_state.analysis_result = (
                        None
                    )

                    st.session_state.canvas_version += (
                        1
                    )

                    st.rerun()

            with analyse_column:
                if st.button(
                    "Analyse Item",
                    type="primary",
                    use_container_width=True,
                    key=(
                        "analyse_lifestyle_button"
                    ),
                ):
                    with st.spinner(
                        "Analysing..."
                    ):
                        st.session_state.analysis_result = (
                            analyse_with_current_weather(
                                selected_image,
                                original_image,
                            )
                        )

                    st.rerun()


# The weather card remains at the bottom of the left workflow column before
# and after analysis, keeping the recommendation column focused and balanced.
with left_column:
    displayed_weather = current_weather
    if st.session_state.analysis_result is not None:
        displayed_weather = (
            st.session_state.analysis_result.get("weather")
            or current_weather
        )
    render_compact_weather(
        displayed_weather,
        st.session_state.weather_error,
    )


# ---------------------------------------------------------
# Right-column results
# ---------------------------------------------------------

with right_column:
    if st.session_state.analysis_result is None:
        st.markdown(
            """
            <div class="empty-result">
                Select and analyse one clothing
                item to view the recommendation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        if DEVELOPER_MODE:
            render_results(
                st.session_state.analysis_result,
                original_image,
                st.session_state.selected_image,
            )
        else:
            render_recommendations_only(
                st.session_state.analysis_result
            )
