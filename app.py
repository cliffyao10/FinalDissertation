import base64
import html
import io
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components_v2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.recognition import (
    extract_image_embedding,
    predict_category_with_confidence,
    predict_semantic_colour,
    predict_style,
)
from src.colour_detection import predict_colour_details
from src.garment_taxonomy import broad_garment_category
from src.recommendation import CATEGORY_TO_SLOT, recommend_outfit
from src.wardrobe import (
    DEFAULT_OUTFIT_GROUPS,
    SLOT_LABELS as WARDROBE_SLOT_LABELS,
    SLOTS as WARDROBE_SLOTS,
    add_outfit_group,
    filter_pieces_by_slot,
    find_outfit,
    load_store,
    new_outfit,
    new_piece,
    pieces_for_slot,
    prune_unused_images,
    remove_outfit,
    remove_outfit_slot,
    remove_outfit_group,
    remove_piece,
    save_piece_image,
    save_store,
    set_outfit_slot,
    update_outfit_details,
    update_piece_name,
)
from src.weather import WeatherServiceError, get_city_weather


FRAME_WIDTH = 520
FRAME_HEIGHT = 400
FRAME_BACKGROUND = (250, 248, 244)
USER_PREFERENCES_PATH = Path("data/user_preferences.json")

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
    page_title="Cove",
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
        }

        .app-brand-lockup {
            display: flex;
            flex-direction: column;
            gap: .02rem;
            line-height: 1.05;
        }

        .app-brand-name {
            font-family: ui-rounded, "Segoe UI Variable Display", "Trebuchet MS", sans-serif;
            font-size: 1.22rem;
            font-weight: 720;
            letter-spacing: -.045em;
        }

        .app-brand-tagline {
            color: var(--fashion-muted);
            font-size: .67rem;
            font-weight: 580;
            letter-spacing: .035em;
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
            font-size: 1.65rem;
            line-height: 1;
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

        .weather-neighbourhood {
            position: absolute;
            right: 1%;
            bottom: -1%;
            left: 1%;
            z-index: 4;
            height: 34%;
            color: rgba(70, 80, 82, .64);
        }
        .weather-neighbourhood::after {
            content: "";
            position: absolute;
            right: 0;
            bottom: .25rem;
            left: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(80,88,86,.22) 15% 85%, transparent);
        }

        .weather-house {
            position: absolute;
            bottom: .35rem;
            width: 8.3rem;
            height: 5.8rem;
            border: 1px solid rgba(255,255,255,.34);
            border-radius: 8px 8px 2px 2px;
            background: linear-gradient(145deg, rgba(127,132,132,.48), rgba(93,105,105,.36));
            box-shadow: 0 12px 26px rgba(68,76,76,.11);
        }
        .weather-house.one { left: 5%; }
        .weather-house.two { right: 7%; width: 6.8rem; height: 4.8rem; opacity: .82; }
        .weather-house .roof {
            position: absolute;
            right: -.65rem;
            bottom: calc(100% - .15rem);
            left: -.65rem;
            height: 3rem;
            background: rgba(83, 93, 95, .55);
            clip-path: polygon(50% 0, 100% 100%, 0 100%);
        }
        .weather-house.two .roof { background: rgba(104, 104, 105, .48); }
        .weather-house .chimney {
            position: absolute;
            right: 1.25rem;
            bottom: calc(100% + 1.1rem);
            width: .65rem;
            height: 1.7rem;
            border-radius: 2px 2px 0 0;
            background: rgba(81,89,90,.54);
        }
        .weather-house .chimney::after {
            content: "";
            position: absolute;
            top: -1.1rem;
            left: -.35rem;
            width: 1.25rem;
            height: 1.25rem;
            border-radius: 50%;
            background: rgba(255,255,255,.22);
            filter: blur(4px);
            animation: weather-chimney-smoke 5s ease-in-out infinite;
        }
        .weather-house .lit-window {
            position: absolute;
            top: 1.25rem;
            width: 1.35rem;
            height: 1.65rem;
            border: 2px solid rgba(78,82,81,.28);
            border-radius: 5px 5px 2px 2px;
            background: rgba(255, 218, 143, .88);
            box-shadow: 0 0 19px rgba(255,211,125,.32);
        }
        .weather-house .lit-window.left { left: 1.15rem; }
        .weather-house .lit-window.right { right: 1.15rem; }
        .weather-house .door {
            position: absolute;
            bottom: 0;
            left: calc(50% - .8rem);
            width: 1.6rem;
            height: 2.45rem;
            border-radius: 999px 999px 0 0;
            background: rgba(66,78,77,.5);
        }
        .weather-snow .weather-house .roof::after {
            content: "";
            position: absolute;
            right: 5%;
            bottom: 7%;
            left: 5%;
            height: .55rem;
            border-radius: 50%;
            background: rgba(255,255,255,.78);
            filter: blur(.4px);
        }

        .weather-tree {
            position: absolute;
            right: 31%;
            bottom: .4rem;
            width: .48rem;
            height: 4.7rem;
            border-radius: 999px 999px 2px 2px;
            background: rgba(87,82,68,.45);
        }
        .weather-tree::before,
        .weather-tree::after {
            content: "";
            position: absolute;
            border-radius: 48% 52% 50% 50%;
            background: rgba(91, 119, 99, .68);
            box-shadow: 0 9px 20px rgba(62,85,70,.11);
        }
        .weather-tree::before { right: -.9rem; bottom: 2.5rem; width: 2.8rem; height: 3.4rem; transform: rotate(8deg); }
        .weather-tree::after { right: -1.8rem; bottom: 1.75rem; width: 3rem; height: 2.8rem; transform: rotate(-13deg); }
        .weather-snow .weather-tree::before,
        .weather-snow .weather-tree::after { background: rgba(181,201,191,.72); box-shadow: inset 0 .38rem rgba(255,255,255,.65); }
        .weather-fog .weather-tree,
        .weather-fog .weather-house.two { opacity: .32; }

        .weather-streetlamp {
            position: absolute;
            right: 43%;
            bottom: .35rem;
            width: .28rem;
            height: 5.7rem;
            border-radius: 999px 999px 0 0;
            background: rgba(65,73,75,.52);
        }
        .weather-streetlamp::before {
            content: "";
            position: absolute;
            top: -.4rem;
            left: -.58rem;
            width: 1.45rem;
            height: 1.25rem;
            border: 2px solid rgba(65,73,75,.45);
            border-radius: 50% 50% 45% 45%;
            background: rgba(255,221,147,.92);
            box-shadow: 0 0 28px rgba(255,214,131,.58);
        }

        .weather-clothesline {
            position: absolute;
            bottom: 5.6rem;
            left: 28%;
            width: 8rem;
            height: 2.6rem;
            border-top: 1px solid rgba(72,78,77,.38);
            transform: rotate(2deg);
        }
        .weather-clothesline i {
            position: absolute;
            top: -.05rem;
            width: 1.35rem;
            height: 1.7rem;
            border-radius: 3px 3px 7px 7px;
            background: rgba(239, 186, 168, .78);
            transform-origin: top center;
            animation: weather-laundry-sway 4s ease-in-out infinite alternate;
        }
        .weather-clothesline i:first-child { left: 1.3rem; }
        .weather-clothesline i:last-child { right: 1.6rem; height: 1.35rem; background: rgba(238,224,181,.8); animation-delay: -2s; }
        .weather-rain .weather-clothesline,
        .weather-snow .weather-clothesline,
        .weather-fog .weather-clothesline,
        .weather-thunderstorm .weather-clothesline { display: none; }

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
        .weather-window-status {
            display: flex;
            align-items: center;
            gap: .65rem;
        }
        .weather-favourite-mark {
            color: #b96f78;
            font-size: 1.12rem;
            line-height: 1;
        }

        @keyframes weather-orb-breathe {
            50% { transform: scale(1.035); box-shadow: 0 0 0 31px rgba(255,235,169,.08), 0 24px 62px rgba(235,173,75,.28); }
        }
        @keyframes weather-cloud-float { 50% { transform: translateY(-7px); } }
        @keyframes weather-face-bob { 50% { margin-top: -7px; } }
        @keyframes weather-twinkle { 50% { opacity: .28; transform: rotate(28deg) scale(.72); } }
        @keyframes weather-chimney-smoke { 50% { transform: translate(.25rem,-.35rem) scale(1.18); opacity: .35; } }
        @keyframes weather-laundry-sway { to { transform: rotate(5deg); } }
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
            background: transparent;
        }

        .city-location-pin {
            position: relative;
            display: block;
            width: 1.35rem;
            height: 1.35rem;
            transform: rotate(-45deg);
            border: 1px solid color-mix(in srgb, var(--fashion-strong) 48%, white);
            border-radius: 50% 50% 50% 8%;
            background: linear-gradient(
                145deg,
                color-mix(in srgb, var(--fashion-primary) 76%, white),
                var(--fashion-strong)
            );
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,.45),
                0 5px 11px color-mix(in srgb, var(--fashion-primary) 22%, transparent);
        }

        .city-location-pin::after {
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            width: .43rem;
            height: .43rem;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: rgba(255,250,241,.92);
            box-shadow: 0 0 0 1px rgba(255,255,255,.34);
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
            .weather-house .chimney::after,
            .weather-clothesline i,
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


def load_saved_cities():
    """Read locally persisted favourite cities without failing app startup."""

    try:
        payload = json.loads(USER_PREFERENCES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    raw_cities = payload.get("saved_cities", []) if isinstance(payload, dict) else []
    saved_cities = []
    seen = set()
    for value in raw_cities:
        city = str(value).strip()
        city_key = city.casefold()
        if city and city_key not in seen:
            saved_cities.append(city)
            seen.add(city_key)
    return saved_cities[:6]


def save_favourite_cities(cities):
    """Persist favourite cities atomically in the local data directory."""

    USER_PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = USER_PREFERENCES_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"saved_cities": cities[:6]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(USER_PREFERENCES_PATH)


def toggle_favourite_city(cities, city):
    """Add or remove one city while preserving a compact, stable order."""

    city = city.strip()
    if not city:
        return list(cities)

    matching_index = next(
        (
            index
            for index, saved_city in enumerate(cities)
            if saved_city.casefold() == city.casefold()
        ),
        None,
    )
    updated_cities = list(cities)
    if matching_index is None:
        updated_cities.append(city)
    else:
        updated_cities.pop(matching_index)
    return updated_cities[-6:]


def apply_theme_choice():
    """Keep the chosen mood when the home-page picker is not rendered."""

    choice = st.session_state.get("theme_picker")
    if choice in THEMES:
        st.session_state.theme = choice


def apply_saved_city_choice():
    """Apply a favourite city before Streamlit recreates the city input."""

    saved_city = st.session_state.get("saved_city_choice")
    if not saved_city:
        return
    st.session_state.city = saved_city
    st.session_state.city_input = saved_city
    st.session_state.weather_data = None
    st.session_state.weather_error = None
    st.session_state.weather_refresh_requested = True


def enable_home_dock_dragging():
    """Attach lightweight pointer dragging to the homepage control glass."""

    drag_component = components_v2.component(
        "wardrobe_home_dock_dragger",
        html="<span aria-hidden='true'></span>",
        css=":host { display: none; }",
        js="""
        export default function(component) {
          const doc = component.parentElement.ownerDocument;
          const host = doc.defaultView;
          const storageKey = "wardrobe-weather-dock-position";
          const interactive = "button,input,label,[role='radio'],[role='option'],[data-testid='stPills']";

          function clampPanel(panel, bounds, x, y) {
            const panelRect = panel.getBoundingClientRect();
            const boundsRect = bounds.getBoundingClientRect();
            const currentX = Number(panel.dataset.dockX || 0);
            const currentY = Number(panel.dataset.dockY || 0);
            const baseLeft = panelRect.left - currentX;
            const baseTop = panelRect.top - currentY;
            const inset = 12;
            return {
              x: Math.min(
                boundsRect.right - panelRect.width - inset - baseLeft,
                Math.max(boundsRect.left + inset - baseLeft, x)
              ),
              y: Math.min(
                boundsRect.bottom - panelRect.height - inset - baseTop,
                Math.max(boundsRect.top + inset - baseTop, y)
              )
            };
          }

          function applyPosition(panel, bounds, x, y) {
            const next = clampPanel(panel, bounds, x, y);
            panel.dataset.dockX = String(next.x);
            panel.dataset.dockY = String(next.y);
            panel.style.setProperty("--dock-x", `${next.x}px`);
            panel.style.setProperty("--dock-y", `${next.y}px`);
          }

          function initialise() {
            const panel = doc.querySelector("div[data-testid='stLayoutWrapper']:has(.home-dock-marker)");
            const bounds = doc.querySelector("[data-testid='stMainBlockContainer']");
            if (!panel || !bounds || panel.dataset.dragReady === "true") return;
            panel.dataset.dragReady = "true";

            let saved = {x: 0, y: 0};
            try { saved = JSON.parse(host.localStorage.getItem(storageKey)) || saved; }
            catch (_) {}
            requestAnimationFrame(() => applyPosition(panel, bounds, Number(saved.x || 0), Number(saved.y || 0)));

            panel.addEventListener("pointerdown", (event) => {
              if (event.button !== 0 || event.target.closest(interactive)) return;
              event.preventDefault();
              panel.setPointerCapture(event.pointerId);
              panel.classList.add("is-dragging");
              const startClientX = event.clientX;
              const startClientY = event.clientY;
              const startX = Number(panel.dataset.dockX || 0);
              const startY = Number(panel.dataset.dockY || 0);

              const move = (moveEvent) => {
                applyPosition(
                  panel,
                  bounds,
                  startX + moveEvent.clientX - startClientX,
                  startY + moveEvent.clientY - startClientY
                );
              };
              const finish = () => {
                panel.classList.remove("is-dragging");
                panel.removeEventListener("pointermove", move);
                panel.removeEventListener("pointerup", finish);
                panel.removeEventListener("pointercancel", finish);
                host.localStorage.setItem(storageKey, JSON.stringify({
                  x: Number(panel.dataset.dockX || 0),
                  y: Number(panel.dataset.dockY || 0)
                }));
              };
              panel.addEventListener("pointermove", move);
              panel.addEventListener("pointerup", finish);
              panel.addEventListener("pointercancel", finish);
            });
          }

          initialise();
          new MutationObserver(initialise).observe(doc.body, {childList: true, subtree: true});
        }
        """,
        isolate_styles=True,
    )
    drag_component(
        key="wardrobe_home_dock_dragger",
        height="content",
        width="content",
    )


def initialise_state():
    """Initialise application state."""

    defaults = {
        "app_page": "home",
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
        "outfit_audience": "womenswear",
        "saved_cities": load_saved_cities(),
        "weather_refresh_requested": False,
        "wardrobe_store": load_store(),
        "wardrobe_view": "My outfits",
        "wardrobe_group_filter": "All",
        "wardrobe_piece_filter": "All",
        "wardrobe_selected_outfit_id": None,
        "wardrobe_capture_target": None,
        "wardrobe_capture_return_page": "wardrobe",
        "wardrobe_dialog_open": False,
        "wardrobe_recognition_result": None,
        "wardrobe_recognition_error": None,
        "wardrobe_uploaded_bytes": None,
        "wardrobe_uploaded_name": None,
        "wardrobe_selected_image": None,
        "wardrobe_selection_ready": False,
        "wardrobe_canvas_version": 0,
        "wardrobe_uploader_version": 0,
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
            .location-pill {{
                background: var(--fashion-soft) !important;
                color: var(--fashion-strong) !important;
            }}
            .city-panel-icon {{
                background: transparent !important;
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
            div[data-testid="stSegmentedControl"] {{
                width:100% !important;
                padding:3px !important;
                border:1px solid color-mix(in srgb,var(--fashion-primary) 28%,#ddd4c8) !important;
                border-radius:999px !important;
                background:color-mix(in srgb,var(--fashion-soft) 62%,rgba(255,255,255,.74)) !important;
            }}
            div[data-testid="stSegmentedControl"] > div {{ width:100% !important; gap:0 !important; }}
            div[data-testid="stSegmentedControl"] button {{
                flex:1 1 50% !important;
                min-height:2.65rem !important;
                justify-content:center !important;
                border:0 !important;
                border-radius:999px !important;
                background:transparent !important;
                color:var(--fashion-muted) !important;
                box-shadow:none !important;
                transition:background-color .18s ease,color .18s ease,box-shadow .18s ease !important;
            }}
            div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {{
                background:var(--fashion-strong) !important;
                color:white !important;
                box-shadow:0 5px 14px color-mix(in srgb,var(--fashion-strong) 24%,transparent) !important;
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
                display: grid;
                grid-template-rows: 96px 2rem minmax(3.4rem, auto);
                align-content: start;
                height: 238px;
                box-sizing: border-box;
                padding: 16px 13px 18px;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 15%, #ece9e5);
                border-radius: 24px 24px 12px 24px;
                background: color-mix(in srgb, white 84%, var(--fashion-soft));
                text-align: center;
                box-shadow: 0 8px 22px color-mix(in srgb, var(--fashion-primary) 8%, transparent);
                transition: transform .2s ease, box-shadow .2s ease;
            }}
            .garment-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 14px 28px color-mix(in srgb, var(--fashion-primary) 13%, transparent);
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
            .garment-card-meta {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: .38rem;
                min-width: 0;
            }}
            .garment-card-slot,
            .garment-card-colour {{
                display: inline-flex;
                align-items: center;
                min-width: 0;
                padding: .22rem .48rem;
                border-radius: 999px;
                background: color-mix(in srgb, var(--fashion-soft) 68%, white);
                color: var(--fashion-muted);
                font-size: .68rem;
                line-height: 1;
                white-space: nowrap;
            }}
            .garment-card-colour::before {{
                content: "";
                width: .42rem;
                height: .42rem;
                margin-right: .28rem;
                flex: 0 0 auto;
                border: 1px solid rgba(42, 49, 45, .16);
                border-radius: 38% 62% 46% 54%;
                background: var(--item-colour, var(--fashion-primary));
            }}
            .garment-card-type {{
                display: grid;
                margin: .12rem 0 0;
                place-items: center;
                font-size: .96rem;
                font-weight: 680;
                line-height: 1.35;
            }}

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker) {{
                padding: .55rem .65rem .75rem;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 12%, #eeeae5) !important;
                border-radius: 28px 28px 14px 28px !important;
                background: rgba(255,255,255,.58) !important;
                box-shadow: 0 10px 30px color-mix(in srgb, var(--fashion-primary) 7%, transparent) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker.primary) {{
                border-color: color-mix(in srgb, var(--fashion-primary) 24%, #ece7e1) !important;
                background: linear-gradient(145deg, rgba(255,255,255,.94), color-mix(in srgb, var(--fashion-soft) 58%, var(--fashion-cream))) !important;
                box-shadow: 0 18px 42px color-mix(in srgb, var(--fashion-primary) 13%, transparent) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker.alternative) {{
                background: color-mix(in srgb, rgba(255,255,255,.74) 88%, var(--fashion-blush)) !important;
                box-shadow: 0 8px 24px color-mix(in srgb, var(--fashion-primary) 6%, transparent) !important;
            }}
            .recommendation-panel-marker {{
                width: 2rem;
                height: .22rem;
                margin: .05rem 0 .1rem;
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
                content: "✦";
                color: color-mix(in srgb, var(--fashion-strong) 72%, #cf7b82);
                font-size: .66rem;
            }}
            .match-pill.alternative {{
                background: color-mix(in srgb, var(--fashion-blush) 52%, white);
            }}
            .recommendation-reason {{
                display: flex;
                align-items: flex-start;
                gap: .55rem;
                margin-top: .75rem;
                padding: .72rem .82rem;
                border-top: 1px solid color-mix(in srgb, var(--fashion-primary) 16%, transparent);
                color: var(--fashion-muted);
                font-size: .76rem;
                line-height: 1.5;
            }}
            .recommendation-reason::before {{
                content: "Why";
                flex: 0 0 auto;
                color: var(--fashion-strong);
                font-size: .65rem;
                font-weight: 760;
                letter-spacing: .08em;
                text-transform: uppercase;
            }}
            @media (prefers-reduced-motion: reduce) {{
                .garment-card {{ transition: none; }}
                .garment-card:hover {{ transform: none; }}
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
            .wardrobe-page-heading {{
                display: flex;
                align-items: center;
                gap: .75rem;
                margin: 0;
                padding: .25rem .35rem;
            }}
            .wardrobe-page-heading h1 {{ margin: 0 0 .08rem !important; font-size: 1.62rem; line-height: 1.05; }}
            .wardrobe-page-heading p {{ margin: 0; color: var(--fashion-muted); font-size: .76rem; line-height: 1.25; }}
            .wardrobe-heading-weather {{
                display: inline-flex;
                align-items: center;
                gap: .48rem;
                min-width: 8.4rem;
                margin-left: .35rem;
                padding: .46rem .7rem;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 18%, rgba(255,255,255,.78));
                border-radius: 18px 18px 8px 18px;
                background: color-mix(in srgb, var(--fashion-soft) 38%, rgba(255,255,255,.68));
                box-shadow: inset 0 1px 0 rgba(255,255,255,.78), 0 6px 16px rgba(68,62,55,.06);
            }}
            .wardrobe-heading-weather > span {{ font-size: 1.12rem; line-height: 1; }}
            .wardrobe-heading-weather strong {{ display:block; color:var(--fashion-ink); font-size:.73rem; line-height:1.1; }}
            .wardrobe-heading-weather small {{ display:block; margin-top:.13rem; color:var(--fashion-muted); font-size:.59rem; line-height:1.1; }}
            div[class*="st-key-wardrobe_piece_filter"] [data-testid="stPills"] {{ gap:.42rem; }}
            div[class*="st-key-wardrobe_piece_filter"] button {{
                min-height: 2.25rem !important;
                padding: .42rem .78rem !important;
                border-radius: 16px 16px 7px 16px !important;
            }}
            .wooden-hanger {{ width: 58px; height: 44px; flex: 0 0 auto; filter: drop-shadow(0 5px 7px rgba(99,67,42,.15)); }}
            .wardrobe-outfit-marker,
            .wardrobe-slot-marker,
            .wardrobe-piece-marker {{ height: 0; overflow: hidden; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker) {{
                padding: .75rem .78rem .8rem;
                border: 1px solid rgba(255,255,255,.78) !important;
                border-radius: 24px 24px 11px 24px !important;
                box-shadow: 0 12px 28px color-mix(in srgb, var(--fashion-primary) 10%, transparent) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-blue) {{ background: linear-gradient(145deg, #f8fbfd, #e4eff8) !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-green) {{ background: linear-gradient(145deg, #fbfdfb, #e5f1e8) !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-pink) {{ background: linear-gradient(145deg, #fffafb, #f6e6ec) !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-beige) {{ background: linear-gradient(145deg, #fffdf9, #f1e7da) !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-minimal-white) {{ background: linear-gradient(145deg, #ffffff, #f1f1ef) !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-slot-marker):not(:has(.wardrobe-outfit-marker)) {{
                height: 250px;
                padding: .58rem .58rem .52rem;
                overflow: hidden;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 16%, #e8e6e1) !important;
                border-radius: 23px 23px 10px 23px !important;
                background: rgba(255,255,255,.72) !important;
                box-shadow: 0 8px 20px rgba(58,61,58,.07) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-slot-marker):not(:has(.wardrobe-outfit-marker))
            > div[data-testid="stVerticalBlock"] {{ gap: .38rem; height: 100%; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-slot-marker):not(:has(.wardrobe-outfit-marker))
            [data-testid="stImage"] img {{
                width: 100% !important;
                height: 100px !important;
                object-fit: contain !important;
                border-radius: 17px;
                background: color-mix(in srgb, white 72%, var(--fashion-soft));
            }}
            .wardrobe-slot-label {{
                color: var(--fashion-muted);
                font-size: .68rem;
                font-weight: 720;
                letter-spacing: .08em;
                text-transform: uppercase;
            }}
            .wardrobe-piece-name {{
                height: 2.7rem;
                overflow: hidden;
                color: var(--fashion-ink);
                font-size: .84rem;
                font-weight: 680;
                line-height: 1.35;
            }}
            .wardrobe-icon-preview {{
                display: grid;
                height: 100px;
                place-items: center;
                border-radius: 17px;
                background: linear-gradient(145deg, white, color-mix(in srgb, var(--fashion-blush) 38%, var(--fashion-soft)));
            }}
            .wardrobe-empty-copy {{
                margin-top: .2rem;
                color: var(--fashion-muted);
                font-size: .75rem;
                text-align: center;
            }}
            .wardrobe-empty-hanger {{
                display: grid;
                min-height: 72px;
                place-items: end center;
                padding-bottom: .35rem;
            }}
            .wardrobe-empty-hanger .wooden-hanger {{ width: 68px; height: 50px; opacity: .9; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-empty-marker):not(:has(.wardrobe-outfit-marker))
            .stButton > button {{
                width: 54px !important;
                height: 42px !important;
                min-height: 42px !important;
                margin: 0 auto !important;
                padding: 0 !important;
                border: 1px solid rgba(116,78,47,.2) !important;
                border-radius: 18px 18px 9px 18px !important;
                background: linear-gradient(145deg, #e4c8a5, #c99e72) !important;
                color: var(--fashion-strong) !important;
                font-size: 1.55rem;
                font-weight: 400;
                box-shadow: 0 8px 16px rgba(107,72,45,.16) !important;
            }}
            div[class*="st-key-wardrobe_add_"] button {{
                width: 54px !important;
                height: 42px !important;
                min-height: 42px !important;
                margin: 0 auto !important;
                padding: 0 !important;
                border: 1px solid rgba(116,78,47,.2) !important;
                border-radius: 18px 18px 9px 18px !important;
                background: linear-gradient(145deg, #e4c8a5, #c99e72) !important;
                color: var(--fashion-strong) !important;
                font-size: 1.55rem !important;
                font-weight: 400 !important;
                box-shadow: 0 8px 16px rgba(107,72,45,.16) !important;
            }}
            div[class*="st-key-new_empty_outfit"] button,
            div[class*="st-key-wardrobe_add_piece"] button {{ border-radius: 18px 18px 8px 18px !important; }}
            div[class*="st-key-wardrobe_add_piece"] {{ display:flex; justify-content:flex-end; }}
            div[class*="st-key-wardrobe_add_piece"] button {{
                width: 108px !important;
                min-width: 108px !important;
                min-height: 42px !important;
                padding: .48rem .65rem !important;
                font-size: .72rem !important;
            }}
            div[class*="st-key-new_empty_outfit"] button {{
                background: color-mix(in srgb, var(--fashion-cream) 76%, white) !important;
                border-color: color-mix(in srgb, var(--fashion-primary) 20%, #ddd6cd) !important;
            }}
            div[class*="st-key-open_wardrobe_button"] button {{
                min-height: 56px !important;
                padding: .75rem .3rem !important;
                font-size: .68rem !important;
                letter-spacing: .035em !important;
                white-space: nowrap !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-piece-marker) {{
                min-height: 330px;
                padding: .8rem;
                border-radius: 24px 24px 11px 24px !important;
                background: rgba(255,255,255,.7) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-piece-marker)
            [data-testid="stImage"] img {{ height: 190px !important; object-fit: contain !important; }}
            .wardrobe-outfit-preview-marker,
            .wardrobe-picker-marker {{ height: 0; overflow: hidden; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker):not(:has(.wardrobe-outfit-marker)) {{
                height: 92px;
                padding: .38rem .42rem !important;
                overflow: hidden;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 13%, #e6e1d9) !important;
                border-radius: 17px 17px 7px 17px !important;
                background: rgba(255,255,255,.58) !important;
                box-shadow: none !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker):not(:has(.wardrobe-outfit-marker))
            [data-testid="stImage"] img {{ width:100% !important; height:55px !important; object-fit:contain !important; }}
            .wardrobe-outfit-preview-icon {{ display:grid; height:55px; place-items:center; }}
            .wardrobe-outfit-preview-icon svg {{ width:46px; height:46px; }}
            .wardrobe-outfit-preview-empty {{ display:grid; height:55px; place-items:center; color:var(--fashion-muted); font-size:1.05rem; opacity:.6; }}
            .wardrobe-outfit-preview-label {{ color:var(--fashion-muted); font-size:.55rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; white-space:nowrap; overflow:hidden; }}
            .wardrobe-group-chip {{ display:inline-flex; margin-top:.1rem; padding:.2rem .52rem; border-radius:999px; background:color-mix(in srgb,var(--fashion-primary) 10%,white); color:var(--fashion-strong); font-size:.62rem; font-weight:680; }}
            .wardrobe-card-meta {{ margin-left:.45rem; color:var(--fashion-muted); font-size:.62rem; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-picker-marker) {{
                min-height: 132px;
                padding: .55rem !important;
                border-radius: 18px 18px 8px 18px !important;
                background: color-mix(in srgb, var(--fashion-soft) 42%, white) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-picker-marker) [data-testid="stImage"] img {{ height:112px !important; object-fit:contain !important; }}
            .wardrobe-empty-state {{
                padding: 3.4rem 1.2rem;
                border: 1px dashed color-mix(in srgb, var(--fashion-primary) 38%, #d7d7d2);
                border-radius: 30px 30px 14px 30px;
                background: rgba(255,255,255,.55);
                color: var(--fashion-muted);
                text-align: center;
            }}
            .style-discovery {{
                margin: .85rem 0 .25rem;
                padding: .72rem;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 13%, rgba(255,255,255,.78));
                border-radius: 26px 26px 12px 26px;
                background: linear-gradient(135deg, rgba(255,255,255,.72), color-mix(in srgb, var(--fashion-cream) 72%, var(--fashion-soft)));
            }}
            .style-discovery-kicker {{ color: var(--fashion-strong); font-size: .62rem; font-weight: 760; letter-spacing: .13em; text-transform: uppercase; }}
            .style-discovery-head {{ display:flex; justify-content:space-between; gap:1rem; align-items:end; margin:.25rem 0 .75rem; }}
            .style-discovery-head h3 {{ margin:0 !important; font-size:1.08rem; }}
            .style-discovery-head span {{ color:var(--fashion-muted); font-size:.7rem; }}
            .style-discovery-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.65rem; }}
            .style-discovery-card {{ min-height:112px; padding:.75rem; border-radius:19px 19px 8px 19px; background:rgba(255,255,255,.66); }}
            .style-discovery-swatch {{ width:44px; height:30px; margin-bottom:.55rem; border-radius:15px 15px 6px 15px; background:linear-gradient(125deg,var(--fashion-blush),var(--fashion-primary)); opacity:.68; }}
            .style-discovery-card:nth-child(2) .style-discovery-swatch {{ background:linear-gradient(125deg,#d7b98e,#f0dfc5); }}
            .style-discovery-card:nth-child(3) .style-discovery-swatch {{ background:linear-gradient(125deg,var(--fashion-soft),#c8b7b0); }}
            .style-discovery-card strong {{ display:block; color:var(--fashion-ink); font-size:.78rem; }}
            .style-discovery-card p {{ margin:.2rem 0 0; color:var(--fashion-muted); font-size:.67rem; line-height:1.35; }}
            /* Wardrobe surfaces use the same warm, theme-aware liquid glass language. */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-detail-marker),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-piece-marker),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-picker-marker) {{
                background: linear-gradient(145deg, rgba(255,255,255,.52), color-mix(in srgb, var(--fashion-soft) 24%, rgba(255,255,255,.30))) !important;
                border: 1px solid rgba(255,255,255,.74) !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.85), 0 16px 38px color-mix(in srgb, var(--fashion-primary) 11%, transparent) !important;
                backdrop-filter: blur(22px) saturate(1.2);
                -webkit-backdrop-filter: blur(22px) saturate(1.2);
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-blue),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-green),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-pink),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-beige),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-marker.mood-minimal-white) {{
                background: linear-gradient(145deg, rgba(255,255,255,.58), color-mix(in srgb, var(--fashion-soft) 34%, rgba(255,255,255,.24))) !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker):not(:has(.wardrobe-outfit-marker)):not(:has(.wardrobe-detail-marker)) {{
                position: relative;
                height: 124px;
                padding: .42rem !important;
                overflow: hidden;
                background: linear-gradient(145deg, rgba(255,255,255,.46), color-mix(in srgb, var(--fashion-soft) 26%, rgba(255,255,255,.28))) !important;
                border: 1px solid rgba(255,255,255,.78) !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.84), 0 8px 20px rgba(65,62,57,.07) !important;
                backdrop-filter: blur(16px) saturate(1.15);
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.large):not(:has(.wardrobe-detail-marker)) {{ height: 350px; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker)
            [data-testid="stImage"] img {{ height: 82px !important; object-fit: contain !important; border-radius: 13px !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.large)
            [data-testid="stImage"] img {{ height: 280px !important; }}
            .wardrobe-outfit-preview-icon {{ height:82px; }}
            .wardrobe-outfit-preview-icon svg {{ width:62px; height:62px; }}
            .wardrobe-outfit-preview-marker.large ~ .wardrobe-outfit-preview-label {{ font-size:.66rem; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.large)
            .wardrobe-outfit-preview-icon {{ height:280px; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.large)
            .wardrobe-outfit-preview-icon svg {{ width:150px; height:150px; }}
            .wardrobe-outfit-preview-empty {{ height:66px; }}
            .wardrobe-outfit-preview-empty .wooden-hanger {{ width:58px; height:44px; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.large)
            .wardrobe-outfit-preview-empty {{ height:270px; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.large)
            .wardrobe-outfit-preview-empty .wooden-hanger {{ width:130px; height:96px; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.photo)
            [data-testid="stPopover"],
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.icon)
            .stButton,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.icon)
            [data-testid="stPopover"] {{ opacity:0; transform:translateY(4px); transition:opacity .18s ease,transform .18s ease; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker):hover
            [data-testid="stPopover"],
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker):hover
            .stButton {{ opacity:1; transform:translateY(0); }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker)
            .stButton button,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker)
            [data-testid="stPopover"] button {{ min-height:28px !important; padding:.25rem .48rem !important; font-size:.62rem !important; border-radius:12px 12px 6px 12px !important; }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-outfit-preview-marker.empty)
            .stButton button,
            div[class*="st-key-wardrobe_add_piece"] button {{
                color: white !important;
                background: linear-gradient(145deg, color-mix(in srgb, var(--fashion-primary) 82%, white), var(--fashion-strong)) !important;
                border: 1px solid color-mix(in srgb, var(--fashion-primary) 62%, white) !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.35), 0 9px 22px color-mix(in srgb, var(--fashion-primary) 28%, transparent) !important;
            }}
            .wardrobe-detail-heading {{ padding:.55rem .4rem .8rem; }}
            .wardrobe-detail-heading span {{ color:var(--fashion-strong); font-size:.66rem; font-weight:760; letter-spacing:.1em; text-transform:uppercase; }}
            .wardrobe-detail-heading h1 {{ margin:.12rem 0 .1rem !important; font-size:2rem; }}
            .wardrobe-detail-heading p {{ margin:0; color:var(--fashion-muted); }}
            div[role="dialog"],
            [data-baseweb="popover"] > div {{
                background: color-mix(in srgb, var(--fashion-surface) 68%, transparent) !important;
                border: 1px solid rgba(255,255,255,.78) !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.82), 0 24px 64px rgba(47,46,43,.18) !important;
                backdrop-filter: blur(28px) saturate(1.22) !important;
                -webkit-backdrop-filter: blur(28px) saturate(1.22) !important;
            }}
            .designer-carousel {{ position:relative; min-height:146px; overflow:hidden; border-radius:19px 19px 8px 19px; }}
            .designer-slide {{
                position:absolute; inset:0; display:grid; grid-template-columns:minmax(160px,31%) 1fr; gap:.7rem;
                padding:.42rem; opacity:0; pointer-events:none;
                background:linear-gradient(135deg,rgba(255,255,255,.50),color-mix(in srgb,var(--fashion-soft) 30%,rgba(255,255,255,.22)));
                animation:designerFade 36s infinite; animation-delay:calc(var(--slide) * 6s);
            }}
            .designer-slide img {{ width:100%; height:126px; object-fit:cover; border-radius:15px 15px 6px 15px; }}
            .designer-slide > div {{ display:flex; flex-direction:column; justify-content:center; align-items:flex-start; padding:.42rem .65rem; }}
            .designer-slide span {{ color:var(--fashion-strong); font-size:.62rem; font-weight:760; letter-spacing:.11em; text-transform:uppercase; }}
            .designer-slide h4 {{ margin:.2rem 0 .08rem; color:var(--fashion-ink); font-size:1.12rem; }}
            .designer-slide p {{ margin:0 0 .55rem; color:var(--fashion-muted); font-size:.72rem; }}
            .designer-slide button {{ padding:.38rem .68rem; border:1px solid rgba(255,255,255,.8); border-radius:999px; color:var(--fashion-strong); background:rgba(255,255,255,.38); font-size:.67rem; }}
            .designer-progress {{ height:2px; margin:.65rem .4rem 0; overflow:hidden; background:rgba(255,255,255,.46); }}
            .designer-progress i {{ display:block; width:100%; height:100%; transform-origin:left; background:var(--fashion-primary); animation:designerProgress 6s linear infinite; }}
            .designer-disclosure {{ margin:.45rem .4rem 0; color:var(--fashion-muted); font-size:.62rem; }}
            .style-discovery-recommendation {{ margin:.7rem 0 !important; padding:.58rem !important; }}
            .style-discovery-recommendation .style-discovery-head {{ display:block; margin:.18rem 0 .48rem; }}
            .style-discovery-recommendation .style-discovery-head h3 {{ font-size:.9rem; }}
            .style-discovery-recommendation .style-discovery-head span {{ display:none; }}
            .style-discovery-recommendation .designer-carousel {{ min-height:128px; }}
            .style-discovery-recommendation .designer-slide {{ grid-template-columns:minmax(118px,38%) 1fr; gap:.48rem; padding:.34rem; }}
            .style-discovery-recommendation .designer-slide img {{ height:110px; }}
            .style-discovery-recommendation .designer-slide > div {{ padding:.25rem .35rem; }}
            .style-discovery-recommendation .designer-slide h4 {{ font-size:.88rem; }}
            .style-discovery-recommendation .designer-slide p {{ margin-bottom:.32rem; font-size:.64rem; }}
            .style-discovery-recommendation .designer-slide button {{ padding:.28rem .5rem; font-size:.59rem; }}
            .style-discovery-recommendation .designer-progress {{ margin-top:.42rem; }}
            .style-discovery-recommendation .designer-disclosure {{ display:none; }}
            @keyframes designerFade {{ 0%,18% {{opacity:0}} 2%,15% {{opacity:1}} 100% {{opacity:0}} }}
            @keyframes designerProgress {{ from {{transform:scaleX(0)}} to {{transform:scaleX(1)}} }}
            /* Streamlit 1.5x container structure: target only the block whose
               direct element child owns the marker, never an ancestor card. */
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker) {{
                padding:.75rem !important;
                border:1px solid rgba(255,255,255,.76) !important;
                border-radius:24px 24px 11px 24px !important;
                background:linear-gradient(145deg,rgba(255,255,255,.50),color-mix(in srgb,var(--fashion-soft) 28%,rgba(255,255,255,.24))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.88),0 16px 38px color-mix(in srgb,var(--fashion-primary) 12%,transparent) !important;
                backdrop-filter:blur(22px) saturate(1.2);
                -webkit-backdrop-filter:blur(22px) saturate(1.2);
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker) {{
                height:380px !important;
                min-height:380px !important;
                max-height:380px !important;
                overflow:hidden;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            [data-testid="stImage"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            [data-testid="stImageContainer"] {{
                width:100% !important;
                height:216px !important;
                overflow:hidden;
                border-radius:18px 18px 9px 18px !important;
                background:color-mix(in srgb,var(--fashion-primary) 5%,#f4eee5) !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            > div:has([data-testid="stImage"]),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            > div:has([data-testid="stImage"]),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker)
            > div:has([data-testid="stImage"]) {{ width:100% !important; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            [data-testid="stFullScreenFrame"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            [data-testid="stFullScreenFrame"] > div,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            [data-testid="stFullScreenFrame"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            [data-testid="stFullScreenFrame"] > div,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker)
            [data-testid="stFullScreenFrame"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker)
            [data-testid="stFullScreenFrame"] > div {{ width:100% !important; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            [data-testid="stImage"] img {{
                width:100% !important;
                height:216px !important;
                object-fit:cover !important;
                object-position:center !important;
                border-radius:18px 18px 9px 18px !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker)
            .wardrobe-piece-name {{
                min-height:2.5rem;
                display:-webkit-box;
                overflow:hidden;
                -webkit-box-orient:vertical;
                -webkit-line-clamp:2;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker) {{
                position:relative;
                gap:.2rem !important;
                height:132px !important;
                min-height:132px !important;
                max-height:132px !important;
                padding:.42rem !important;
                overflow:hidden;
                border:1px solid rgba(255,255,255,.80) !important;
                border-radius:17px 17px 7px 17px !important;
                background:linear-gradient(145deg,rgba(255,255,255,.48),color-mix(in srgb,var(--fashion-soft) 32%,rgba(255,255,255,.22))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.9),0 8px 18px rgba(65,62,57,.07) !important;
                backdrop-filter:blur(16px) saturate(1.14);
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            .wardrobe-outfit-preview-label {{
                position:absolute;
                z-index:3;
                top:.58rem;
                left:.68rem;
                right:.68rem;
                line-height:1;
                pointer-events:none;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.large) {{
                height:350px !important;
                min-height:350px !important;
                max-height:350px !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            [data-testid="stImage"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            [data-testid="stImageContainer"] {{
                width:100% !important;
                height:88px !important;
                margin-top:1.05rem !important;
                overflow:hidden;
                border-radius:12px !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            [data-testid="stImage"] img {{ width:100% !important; height:88px !important; object-fit:cover !important; object-position:center !important; border-radius:12px !important; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.large)
            [data-testid="stImage"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.large)
            [data-testid="stImageContainer"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.large)
            [data-testid="stImage"] img {{ height:286px !important; object-fit:contain !important; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker):not(:has(.large))
            .wardrobe-outfit-preview-icon,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker):not(:has(.large))
            .wardrobe-outfit-preview-empty {{ height:88px !important; margin-top:1.05rem; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.large)
            .wardrobe-outfit-preview-icon {{ height:286px; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.large)
            .wardrobe-outfit-preview-empty {{ height:272px; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker)
            [data-testid="stImage"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker)
            [data-testid="stImageContainer"] {{ width:100% !important; height:138px !important; overflow:hidden; border-radius:15px !important; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker)
            [data-testid="stImage"] img {{
                width:100% !important;
                height:138px !important;
                object-fit:cover !important;
                object-position:center !important;
                border-radius:15px !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            > div:has(.stButton),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            > div:has([data-testid="stPopover"]) {{ position:absolute; z-index:4; bottom:.38rem; width:auto !important; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            > div:has(.stButton) {{ left:.38rem; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            > div:has([data-testid="stPopover"]) {{ right:.38rem; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.empty)
            > div:has(.stButton) {{ left:.38rem; right:.38rem; }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.photo)
            > div:has([data-testid="stPopover"]),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.empty)
            > div:has(.stButton),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.icon)
            > div:has(.stButton),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.icon)
            > div:has([data-testid="stPopover"]) {{
                opacity:0;
                visibility:hidden;
                pointer-events:none;
                transform:translateY(4px);
                transition:opacity .18s ease,transform .18s ease,visibility 0s linear .18s;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker):hover
            > div:has(.stButton),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker):hover
            > div:has([data-testid="stPopover"]) {{
                opacity:1;
                visibility:visible;
                pointer-events:auto;
                transform:translateY(0);
                transition-delay:0s;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            .stButton button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker)
            [data-testid="stPopover"] button {{
                width:auto !important; min-width:58px !important; min-height:29px !important; padding:.25rem .48rem !important;
                border:1px solid rgba(255,255,255,.8) !important; border-radius:12px 12px 6px 12px !important;
                background:color-mix(in srgb,var(--fashion-surface) 62%,transparent) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.85),0 6px 16px rgba(46,44,41,.12) !important;
                backdrop-filter:blur(16px); white-space:nowrap !important; font-size:.62rem !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.empty)
            .stButton button,
            div[class*="st-key-wardrobe_add_piece"] button {{
                color:white !important;
                background:linear-gradient(145deg,color-mix(in srgb,var(--fashion-primary) 82%,white),var(--fashion-strong)) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 58%,white) !important;
            }}
            /* Theme-aware liquid-glass controls across wardrobe cards, detail view and overlays. */
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            .stButton > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            [data-testid="stPopover"] > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            .stButton > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            [data-testid="stPopover"] > button,
            div[role="dialog"] .stButton > button,
            [data-baseweb="popover"] .stButton > button {{
                color:var(--fashion-strong) !important;
                border:1px solid color-mix(in srgb,var(--fashion-primary) 24%,rgba(255,255,255,.86)) !important;
                border-radius:15px 15px 7px 15px !important;
                background:linear-gradient(145deg,
                    color-mix(in srgb,var(--fashion-primary) 17%,rgba(255,255,255,.60)),
                    color-mix(in srgb,var(--fashion-soft) 45%,rgba(255,255,255,.30))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.88),
                    0 8px 20px color-mix(in srgb,var(--fashion-primary) 14%,transparent) !important;
                backdrop-filter:blur(18px) saturate(1.22) !important;
                -webkit-backdrop-filter:blur(18px) saturate(1.22) !important;
                transition:transform .18s ease,background .18s ease,box-shadow .18s ease !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            .stButton > button:hover,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            [data-testid="stPopover"] > button:hover,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            .stButton > button:hover,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            [data-testid="stPopover"] > button:hover,
            div[role="dialog"] .stButton > button:hover,
            [data-baseweb="popover"] .stButton > button:hover {{
                color:var(--fashion-strong) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 42%,white) !important;
                background:linear-gradient(145deg,
                    color-mix(in srgb,var(--fashion-primary) 28%,rgba(255,255,255,.58)),
                    color-mix(in srgb,var(--fashion-soft) 56%,rgba(255,255,255,.28))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.92),
                    0 11px 25px color-mix(in srgb,var(--fashion-primary) 22%,transparent) !important;
                transform:translateY(-1px);
            }}
            div[class*="st-key-save_outfit_details_"] button,
            div[class*="st-key-save_wardrobe_piece"] button,
            div[class*="st-key-wardrobe_add_piece"] button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker.empty)
            .stButton button {{
                color:var(--fashion-strong) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 28%,rgba(255,255,255,.82)) !important;
                background:linear-gradient(145deg,
                    color-mix(in srgb,var(--fashion-primary) 20%,rgba(255,255,255,.64)),
                    color-mix(in srgb,var(--fashion-soft) 36%,rgba(255,255,255,.30))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.90),
                    0 9px 22px color-mix(in srgb,var(--fashion-primary) 15%,transparent) !important;
            }}
            div[class*="st-key-delete_outfit_"] button:not(:disabled),
            div[class*="st-key-outfit_slot_"][class*="confirm_remove"] button {{
                color:#8e3636 !important;
                border-color:rgba(181,78,78,.30) !important;
                background:linear-gradient(145deg,rgba(255,235,235,.66),rgba(222,132,132,.20)) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.82),0 8px 20px rgba(145,60,60,.13) !important;
            }}
            div[role="dialog"] .stButton > button:disabled,
            [data-baseweb="popover"] .stButton > button:disabled {{
                opacity:.48;
                box-shadow:none !important;
                transform:none !important;
            }}
            /* Warm opaline glass: cream diffusion and a faint amber inner glow. */
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker),
            .style-discovery {{
                border-color:rgba(255,246,229,.88) !important;
                background:
                    radial-gradient(circle at 16% 0%,rgba(255,229,190,.34),transparent 34%),
                    linear-gradient(145deg,
                        rgba(255,251,242,.64),
                        color-mix(in srgb,var(--fashion-primary) 9%,rgba(244,229,207,.48))) !important;
                box-shadow:
                    inset 0 1px 0 rgba(255,255,255,.94),
                    inset 0 -1px 0 rgba(173,122,70,.07),
                    0 14px 34px rgba(105,76,48,.09) !important;
                backdrop-filter:blur(17px) saturate(1.08) !important;
                -webkit-backdrop-filter:blur(17px) saturate(1.08) !important;
            }}
            div[role="dialog"],
            [data-baseweb="popover"] > div {{
                border-color:rgba(255,246,229,.9) !important;
                background:
                    radial-gradient(circle at 12% 0%,rgba(255,226,181,.30),transparent 32%),
                    color-mix(in srgb,#fff8ec 72%,transparent) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.94),
                    inset 0 -1px 0 rgba(171,117,63,.08),
                    0 24px 60px rgba(91,63,39,.15) !important;
                backdrop-filter:blur(24px) saturate(1.08) !important;
                -webkit-backdrop-filter:blur(24px) saturate(1.08) !important;
            }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker) {{
                width:min(310px,calc(100vw - 2rem)) !important;
                min-width:0 !important;
                max-width:310px !important;
            }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker)
            div[data-testid="stVerticalBlock"] {{ gap:.42rem !important; }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker)
            .outfit-manage-popover-marker {{ height:0; overflow:hidden; }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker)
            [data-testid="stWidgetLabel"] p,
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker)
            [data-testid="stCaptionContainer"] p {{ font-size:.72rem !important; }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker)
            input {{ min-height:38px !important; height:38px !important; }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker)
            .stButton > button {{ min-height:38px !important; padding:.42rem .72rem !important; }}
            [data-testid="stPopoverBody"]:has(.outfit-manage-popover-marker) hr {{ margin:.45rem 0 !important; }}
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            .stButton > button,
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            [data-testid="stPopover"] > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            .stButton > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            [data-testid="stPopover"] > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            .stButton > button,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            [data-testid="stPopover"] > button,
            div[role="dialog"] .stButton > button,
            [data-baseweb="popover"] .stButton > button {{
                border-color:color-mix(in srgb,var(--fashion-primary) 18%,rgba(255,246,229,.92)) !important;
                background:
                    radial-gradient(circle at 18% 5%,rgba(255,238,208,.48),transparent 44%),
                    linear-gradient(145deg,
                        rgba(255,251,242,.58),
                        color-mix(in srgb,var(--fashion-primary) 13%,rgba(245,230,209,.38))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.94),
                    inset 0 -1px 0 rgba(166,111,59,.07),
                    0 7px 18px rgba(103,73,46,.09) !important;
                backdrop-filter:blur(14px) saturate(1.06) !important;
                -webkit-backdrop-filter:blur(14px) saturate(1.06) !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            .stButton > button:hover,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker)
            [data-testid="stPopover"] > button:hover,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            .stButton > button:hover,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker)
            [data-testid="stPopover"] > button:hover,
            div[role="dialog"] .stButton > button:hover,
            [data-baseweb="popover"] .stButton > button:hover {{
                border-color:color-mix(in srgb,var(--fashion-primary) 30%,#fff2dc) !important;
                background:
                    radial-gradient(circle at 18% 5%,rgba(255,231,190,.56),transparent 44%),
                    linear-gradient(145deg,
                        rgba(255,250,239,.68),
                        color-mix(in srgb,var(--fashion-primary) 19%,rgba(244,224,197,.42))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.96),
                    inset 0 -1px 0 rgba(166,111,59,.08),
                    0 10px 22px rgba(103,73,46,.13) !important;
            }}
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            .stButton > button,
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            [data-testid="stPopover"] > button {{
                color:var(--fashion-strong) !important;
                border:1px solid color-mix(in srgb,var(--fashion-primary) 18%,rgba(255,246,229,.92)) !important;
                border-radius:15px 15px 7px 15px !important;
                background:
                    radial-gradient(circle at 18% 5%,rgba(255,238,208,.48),transparent 44%),
                    linear-gradient(145deg,
                        rgba(255,251,242,.58),
                        color-mix(in srgb,var(--fashion-primary) 13%,rgba(245,230,209,.38))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.94),
                    inset 0 -1px 0 rgba(166,111,59,.07),
                    0 7px 18px rgba(103,73,46,.09) !important;
                backdrop-filter:blur(14px) saturate(1.06) !important;
                -webkit-backdrop-filter:blur(14px) saturate(1.06) !important;
            }}
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            .stButton > button:hover,
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            [data-testid="stPopover"] > button:hover {{
                color:var(--fashion-strong) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 30%,#fff2dc) !important;
                background:
                    radial-gradient(circle at 18% 5%,rgba(255,231,190,.56),transparent 44%),
                    linear-gradient(145deg,
                        rgba(255,250,239,.68),
                        color-mix(in srgb,var(--fashion-primary) 19%,rgba(244,224,197,.42))) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.96),
                    inset 0 -1px 0 rgba(166,111,59,.08),
                    0 10px 22px rgba(103,73,46,.13) !important;
                transform:translateY(-1px);
            }}
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            div[class*="st-key-delete_outfit_"] button:not(:disabled),
            [data-testid="stMainBlockContainer"]:has(.wardrobe-page-heading)
            div[class*="st-key-clear_all_outfits"] button:not(:disabled),
            div[class*="st-key-outfit_slot_"][class*="confirm_remove"] button {{
                color:#8e4a42 !important;
                border-color:rgba(181,99,84,.25) !important;
                background:linear-gradient(145deg,rgba(255,244,237,.62),rgba(222,142,122,.16)) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.9),0 7px 18px rgba(145,76,60,.10) !important;
            }}
            /* Grounded textile glass: warm, tactile and deliberately non-floating. */
            div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker),
            .garment-card,
            .style-discovery {{
                box-shadow:none !important;
                transform:none !important;
            }}
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker),
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker) {{
                border-color:rgba(181,153,115,.25) !important;
                background-color:rgba(251,245,233,.78) !important;
                background-image:
                    repeating-linear-gradient(45deg,rgba(126,94,59,.025) 0 1px,transparent 1px 5px),
                    repeating-linear-gradient(-45deg,rgba(255,255,255,.20) 0 1px,transparent 1px 6px) !important;
                backdrop-filter:blur(12px) saturate(1.02) !important;
                -webkit-backdrop-filter:blur(12px) saturate(1.02) !important;
            }}
            .wooden-hanger {{ filter:none !important; }}
            .garment-card {{
                border-color:rgba(181,153,115,.23) !important;
                background:rgba(252,247,238,.86) !important;
                transition:border-color .18s ease,background-color .18s ease !important;
            }}
            .garment-card:hover {{
                border-color:color-mix(in srgb,var(--fashion-primary) 30%,#d8c5aa) !important;
                background:color-mix(in srgb,var(--fashion-primary) 5%,#fcf7ee) !important;
                box-shadow:none !important;
                transform:none !important;
            }}
            .stButton > button,
            .stFormSubmitButton > button,
            [data-testid="stPopover"] > button {{
                color:var(--fashion-strong) !important;
                border:1px solid rgba(174,145,108,.30) !important;
                border-radius:15px 15px 7px 15px !important;
                background:rgba(252,247,237,.78) !important;
                box-shadow:none !important;
                filter:none !important;
                transform:none !important;
                backdrop-filter:blur(9px) !important;
                -webkit-backdrop-filter:blur(9px) !important;
                transition:background-color .18s ease,border-color .18s ease,color .18s ease !important;
            }}
            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {{
                color:var(--fashion-strong) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 34%,#d9c7ac) !important;
                background:color-mix(in srgb,var(--fashion-primary) 13%,rgba(252,247,237,.84)) !important;
                box-shadow:none !important;
            }}
            .stButton > button:hover,
            .stFormSubmitButton > button:hover,
            [data-testid="stPopover"] > button:hover,
            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button[kind="primary"]:hover {{
                color:white !important;
                border-color:var(--fashion-strong) !important;
                background:var(--fashion-strong) !important;
                box-shadow:none !important;
                filter:none !important;
                transform:none !important;
            }}
            .stButton > button:disabled,
            .stFormSubmitButton > button:disabled {{
                color:var(--fashion-muted) !important;
                border-color:rgba(174,145,108,.16) !important;
                background:rgba(245,239,228,.48) !important;
                opacity:.58;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker.primary),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.recommendation-panel-marker.alternative) {{
                border-color:rgba(181,153,115,.24) !important;
                background:rgba(252,247,238,.70) !important;
                box-shadow:none !important;
            }}
            @media (max-width: 760px) {{
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-slot-marker):not(:has(.wardrobe-outfit-marker)) {{ height: 240px; }}
                .wardrobe-page-heading h1 {{ font-size: 1.7rem; }}
                .style-discovery-grid {{ grid-template-columns:1fr; }}
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
    st.session_state.app_page = "recommend"

    clear_uploaded_image()


def return_to_mode_selection():
    """Return to the opening page."""

    st.session_state.image_mode = None
    st.session_state.app_page = "home"

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
        audience=st.session_state.get("outfit_audience", "womenswear"),
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
        "audience": st.session_state.get("outfit_audience", "womenswear"),
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
    raw_city = str((weather or {}).get("city", st.session_state.city))
    city = html.escape(raw_city)
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
        temperature = "—"
        accessibility = "A calm warm window waiting for a city forecast"

    favourite_mark = (
        "♥"
        if any(
            saved_city.casefold() == raw_city.casefold()
            for saved_city in st.session_state.saved_cities
        )
        else "♡"
    )

    st.markdown(
        f"""
        <div class="weather-window-scene weather-{scene}" role="img"
             aria-label="{html.escape(accessibility)}">
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
                <div class="weather-neighbourhood">
                    <div class="weather-house one">
                        <i class="roof"></i><i class="chimney"></i>
                        <i class="lit-window left"></i><i class="lit-window right"></i>
                        <i class="door"></i>
                    </div>
                    <div class="weather-house two">
                        <i class="roof"></i><i class="lit-window left"></i>
                        <i class="lit-window right"></i><i class="door"></i>
                    </div>
                    <div class="weather-tree"></div>
                    <div class="weather-streetlamp"></div>
                    <div class="weather-clothesline"><i></i><i></i></div>
                </div>
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
                <div class="weather-window-status">
                    <div class="weather-window-temperature">{temperature}</div>
                    <span class="weather-favourite-mark">{favourite_mark}</span>
                </div>
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
        "audience": result.get(
            "audience", st.session_state.get("outfit_audience", "womenswear")
        ),
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

    item_name = broad_garment_category(slot, item_type).casefold()
    if "too hot" in item_name or "no outer" in item_name:
        icon_name = "no_layer"
    elif slot == "inner_top" and any(
        name in item_name for name in ("tank", "vest", "cami")
    ):
        icon_name = "tank_top"
    elif slot == "bottom" and "short" in item_name:
        icon_name = "shorts"
    elif slot == "bottom" and "skirt" in item_name:
        icon_name = "skirt"
    elif slot == "inner_top" and "dress" in item_name:
        icon_name = "dress"
    elif slot == "shoes" and "sandal" in item_name:
        icon_name = "sandals"
    elif slot == "shoes" and "boot" in item_name:
        icon_name = "boots"
    elif slot == "shoes" and any(
        name in item_name
        for name in ("loafer", "dress shoe")
    ):
        icon_name = "leather_shoes"
    elif slot == "shoes" and any(
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


def outfit_match_label(outfit, *, alternative=False):
    """Translate an internal ranker score into friendly interface copy."""

    if outfit.get("user_edited"):
        return "Your edit"
    if alternative:
        return "Playful alternative"

    score = outfit.get("model_score")
    if score is None:
        return "Cove pick"
    if float(score) >= 82:
        return "Great match"
    if float(score) >= 68:
        return "Good match"
    return "Fresh pairing"


def wooden_hanger_svg():
    """Return a small, warm wooden hanger illustration for wardrobe surfaces."""

    return (
        '<svg class="wooden-hanger" viewBox="0 0 120 86" aria-hidden="true" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path d="M61 33c-7-1-10-6-9-12 1-7 11-10 16-5 5 5 1 11-4 14-3 2-4 5-4 8" '
        'fill="none" stroke="#8b6748" stroke-width="5" stroke-linecap="round"/>'
        '<path d="M59 37 14 69c-4 3-2 9 3 9h86c5 0 7-6 3-9L61 37Z" '
        'fill="#d8b58c" stroke="#9b714c" stroke-width="4" stroke-linejoin="round"/>'
        '<path d="M28 69h64" stroke="#f1d8b8" stroke-width="3" stroke-linecap="round" opacity=".8"/>'
        '</svg>'
    )


@st.cache_data(show_spinner=False)
def local_image_data_uri(path_string):
    path = Path(path_string)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def render_style_discovery(context):
    """Rotate six original independent-designer concept advertisements."""

    subtitle = (
        "Pieces that could live beside today's recommendation."
        if context == "recommendation"
        else "A future edit shaped by the looks and colours you keep."
    )
    concepts = [
        ("oat-wrap-jacket.webp", "Morrow Atelier", "Hand-finished wrap wool"),
        ("moss-knit-cardigan.webp", "Fern & Fold", "A softer kind of structure"),
        ("dusty-rose-skirt.webp", "Lune Form", "Washed silk in motion"),
        ("ink-rain-cape.webp", "Still Rain", "Weatherwear, quietly redrawn"),
        ("terracotta-blouse.webp", "Ochre Room", "Gathered cotton, made slowly"),
        ("cream-woven-shoes.webp", "Pale Thread", "Woven leather by hand"),
    ]
    slides = []
    for index, (filename, label, description) in enumerate(concepts):
        uri = local_image_data_uri(
            str(Path("assets/designer_discovery") / filename)
        )
        slides.append(
            f'<article class="designer-slide" style="--slide:{index}">'
            f'<img src="{uri}" alt="{html.escape(description)}">'
            '<div><span>Independent designer concept</span>'
            f'<h4>{html.escape(label)}</h4><p>{html.escape(description)}</p>'
            '<button type="button" tabindex="-1">Discover the piece</button></div>'
            '</article>'
        )
    context_class = " style-discovery-recommendation" if context == "recommendation" else ""
    st.markdown(
        f'<section class="style-discovery{context_class}">'
        '<div class="style-discovery-kicker">Style discovery · designer rotation</div>'
        '<div class="style-discovery-head"><h3>A little more from your world</h3>'
        f'<span>{html.escape(subtitle)}</span></div>'
        f'<div class="designer-carousel">{"".join(slides)}</div>'
        '<div class="designer-progress"><i></i></div>'
        '<p class="designer-disclosure">Concept placements for the future personalised discovery model.</p>'
        '</section>',
        unsafe_allow_html=True,
    )


def outfit_reason(outfit, weather):
    """Explain an outfit in short, user-facing language."""

    weather = weather or {}
    feels_like = float(weather.get("feels_like", 20.0))
    rain_probability = float(weather.get("rain_probability", 0.0))
    condition = str(weather.get("condition", "Clear"))

    if rain_probability >= 50 or condition in {"Rain", "Thunderstorm"}:
        weather_note = "Weather-ready choices keep possible rain in mind"
    elif condition == "Snow" or feels_like <= 7:
        weather_note = "Cosy layers suit the colder conditions"
    elif feels_like >= 25:
        weather_note = "Breathable pieces keep the warmer day comfortable"
    else:
        weather_note = "Easy layers suit today's temperature"

    colours = []
    for item in outfit.get("items", []):
        colour = item.get("colour")
        if colour and colour not in colours:
            colours.append(colour)

    if len(colours) >= 2:
        colour_note = f"{colours[0]} and {colours[1]} keep the palette balanced"
    elif colours:
        colour_note = f"A focused {colours[0].lower()} palette keeps it calm"
    else:
        colour_note = "The pieces stay visually balanced"

    return f"{weather_note}. {colour_note}."


def render_outfit_cards(outfit, outfit_key):
    """Render one three-item outfit with coloured category icons."""

    columns = st.columns(3, gap="medium")

    for item_index, (column, item) in enumerate(zip(columns, outfit["items"])):
        with column:
            broad_category = broad_garment_category(
                item["slot"], item.get("type", "")
            )
            colour_line = (
                '<span class="garment-card-colour" '
                f'style="--item-colour:{ICON_COLOURS.get(item["colour"], "#d8d8d3")}">'
                f'{html.escape(item["colour"])}</span>'
                if item.get("colour")
                else ""
            )
            st.markdown(
                (
                    '<div class="garment-card">'
                    '<div class="garment-icon-shell">'
                    f'{garment_icon_svg(item["slot"], item["colour"], broad_category)}'
                    '</div>'
                    '<div class="garment-card-meta">'
                    f'<span class="garment-card-slot">{html.escape(item["slot_label"])}</span>'
                    f'{colour_line}</div>'
                    f'<div class="garment-card-type">{html.escape(broad_category)}</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
            alternatives = outfit.get("alternatives_by_slot", {}).get(
                item["slot"], []
            )
            model_alternatives = [
                {
                    **candidate,
                    "type": broad_garment_category(
                        item["slot"], candidate.get("type", "")
                    ),
                }
                for candidate in alternatives
                if candidate.get("colour")
                and candidate.get("colour") != item.get("colour")
                and candidate.get("selection_source")
            ][:3]
            if model_alternatives and item.get("colour"):
                with st.popover("Model alternatives", use_container_width=True):
                    st.caption(
                        "These are the highest-ranked small changes for this complete outfit."
                    )
                    choice_index = st.selectbox(
                        "Model-ranked option",
                        range(len(model_alternatives)),
                        format_func=lambda index: (
                            f'{model_alternatives[index].get("colour", "")} · '
                            f'{model_alternatives[index].get("type", "Item")} · '
                            f'{model_alternatives[index].get("compatibility_score", 0):.0f}% match'
                        ),
                        key=f'{outfit_key}_{item["slot"]}_model_choice',
                    )
                    if st.button(
                        "Use model choice",
                        key=f'{outfit_key}_{item["slot"]}_apply',
                        type="primary",
                        use_container_width=True,
                    ):
                        replacement = dict(model_alternatives[choice_index])
                        replacement_components = replacement.pop(
                            "score_components", None
                        )
                        outfit["items"][item_index] = replacement
                        outfit["model_score"] = replacement.get(
                            "compatibility_score", outfit.get("model_score")
                        )
                        if replacement_components is not None:
                            outfit["score_components"] = replacement_components
                        outfit["user_edited"] = True
                        st.rerun()


def render_recommendations_only(result):
    """Render recognised styles and two complete recommendations."""

    recommendation = result["recommendation"]
    existing_alternatives = recommendation.get("primary", {}).get(
        "alternatives_by_slot", {}
    )
    has_legacy_manual_alternatives = any(
        candidate and not candidate.get("selection_source")
        for candidates in existing_alternatives.values()
        for candidate in candidates
    )

    # Rebuild results created before the four-slot recommender was loaded.
    if (
        "primary" not in recommendation
        or "alternatives_by_slot" not in recommendation["primary"]
        or has_legacy_manual_alternatives
    ):
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
            audience=result.get(
                "audience", st.session_state.get("outfit_audience", "womenswear")
            ),
        )
        result["recommendation"] = recommendation

    weather = result.get("weather")

    recognised_styles = [
        item["style"]
        for item in result["style_result"].get("styles", [])
    ] or ["Casual"]

    with st.container(border=True):
        st.markdown(
            '<div class="recommendation-panel-marker style"></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Set the mood")
        st.caption(
            "Recognised: " + ", ".join(recognised_styles)
        )

        current_audience = result.get(
            "audience", st.session_state.get("outfit_audience", "womenswear")
        )
        audience_labels = {
            "Menswear": "menswear",
            "Womenswear": "womenswear",
        }
        current_label = next(
            label
            for label, value in audience_labels.items()
            if value == current_audience
        )
        if "recommendation_audience_choice" not in st.session_state:
            st.session_state.recommendation_audience_choice = current_label
        selected_audience_label = st.segmented_control(
            "Clothing range",
            list(audience_labels),
            selection_mode="single",
            key="recommendation_audience_choice",
            label_visibility="collapsed",
        ) or current_label
        selected_audience = audience_labels[selected_audience_label]
        if selected_audience != current_audience:
            st.session_state.outfit_audience = selected_audience
            result["audience"] = selected_audience
            result["recommendation"] = recommend_outfit(
                result["recommendation_category"],
                result["colour"],
                styles=recognised_styles,
                weather=weather,
                selected_style=st.session_state.get("recommendation_style_choice"),
                input_embedding=result.get("input_embedding"),
                audience=selected_audience,
            )
            st.session_state.analysis_result = result
            recommendation = result["recommendation"]

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
                "Try another look ✦",
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
                    audience=selected_audience,
                )
                st.session_state.analysis_result = result
                st.rerun()

        if st.button(
            "❤️ Save this piece and finish the outfit later",
            key="save_uploaded_piece_only",
            use_container_width=True,
        ):
            save_analysis_to_wardrobe(result)

    recommendation = result["recommendation"]

    with st.container(border=True):
        st.markdown(
            '<div class="recommendation-panel-marker primary"></div>',
            unsafe_allow_html=True,
        )
        st.subheader(
            f'Today\'s outfit · {recommendation["primary"]["style"]}'
        )
        st.markdown(
            '<div class="match-pill">'
            f'{outfit_match_label(recommendation["primary"])}</div>',
            unsafe_allow_html=True,
        )
        render_outfit_cards(recommendation["primary"], "primary_outfit")
        st.markdown(
            '<div class="recommendation-reason">'
            f'{html.escape(outfit_reason(recommendation["primary"], weather))}'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "❤️ Save this outfit",
            key="save_primary_outfit",
            use_container_width=True,
        ):
            save_analysis_to_wardrobe(result, recommendation["primary"])

    with st.container(border=True):
        st.markdown(
            '<div class="recommendation-panel-marker alternative"></div>',
            unsafe_allow_html=True,
        )
        st.subheader(
            f'Another idea · {recommendation["alternative"]["style"]}'
        )
        st.markdown(
            '<div class="match-pill alternative">'
            f'{outfit_match_label(recommendation["alternative"], alternative=True)}</div>',
            unsafe_allow_html=True,
        )
        render_outfit_cards(recommendation["alternative"], "alternative_outfit")
        st.markdown(
            '<div class="recommendation-reason">'
            f'{html.escape(outfit_reason(recommendation["alternative"], weather))}'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "❤️ Save this outfit",
            key="save_alternative_outfit",
            use_container_width=True,
        ):
            save_analysis_to_wardrobe(result, recommendation["alternative"])

def persist_wardrobe():
    """Persist the private wardrobe currently held in session state."""

    save_store(st.session_state.wardrobe_store)
    prune_unused_images(st.session_state.wardrobe_store)


def reset_wardrobe_capture():
    st.session_state.wardrobe_uploaded_bytes = None
    st.session_state.wardrobe_uploaded_name = None
    st.session_state.wardrobe_selected_image = None
    st.session_state.wardrobe_selection_ready = False
    st.session_state.wardrobe_capture_target = None
    st.session_state.wardrobe_dialog_open = False
    st.session_state.wardrobe_recognition_result = None
    st.session_state.wardrobe_recognition_error = None
    st.session_state.wardrobe_canvas_version += 1
    st.session_state.wardrobe_uploader_version += 1
    for widget_key in (
        "wardrobe_piece_slot",
        "wardrobe_piece_type",
        "wardrobe_piece_colour",
    ):
        st.session_state.pop(widget_key, None)


def open_wardrobe_capture(outfit_id=None, slot=None):
    return_page = (
        st.session_state.app_page
        if st.session_state.app_page in {"wardrobe", "wardrobe_outfit"}
        else "wardrobe"
    )
    reset_wardrobe_capture()
    st.session_state.wardrobe_capture_target = {
        "outfit_id": outfit_id,
        "slot": slot,
    }
    st.session_state.wardrobe_dialog_open = True
    st.session_state.wardrobe_capture_return_page = return_page
    st.session_state.app_page = return_page


def recognise_wardrobe_piece(selected_image, original_image, target_slot=None):
    """Recognise a wardrobe piece once and seed fields the user can edit."""

    if st.session_state.wardrobe_recognition_result is not None:
        return st.session_state.wardrobe_recognition_result

    try:
        with st.spinner("Cove is recognising your piece…"):
            category_result = predict_category_with_confidence(selected_image)
            colour_result = predict_semantic_colour(selected_image)

        category = category_result.get("predicted_fine_category") or category_result.get(
            "category", ""
        )
        if category_result.get("category_out_of_scope"):
            category = ""

        recognised_colour = colour_result.get("colour", "")
        if recognised_colour not in ICON_COLOURS:
            recognised_colour = ""

        suggested_slot = target_slot or CATEGORY_TO_SLOT.get(
            category,
            CATEGORY_TO_SLOT.get(
                category_result.get("recommendation_category"), "inner_top"
            ),
        )
        result = {
            "category": category,
            "colour": recognised_colour,
            "slot": suggested_slot,
            "category_confidence": float(category_result.get("confidence", 0.0)),
            "colour_confidence": float(colour_result.get("confidence", 0.0)),
        }
        st.session_state.wardrobe_recognition_result = result
        st.session_state.wardrobe_piece_slot = suggested_slot
        st.session_state.wardrobe_piece_type = category
        st.session_state.wardrobe_piece_colour = recognised_colour
        return result
    except Exception as exc:
        # Keep manual wardrobe entry available when the local model cannot load.
        st.session_state.wardrobe_recognition_error = str(exc)
        pixel_result = predict_colour_details(selected_image, reference_image=original_image)
        fallback_colour = pixel_result.get("primary_colour", "")
        if fallback_colour not in ICON_COLOURS:
            fallback_colour = ""
        result = {
            "category": "",
            "colour": fallback_colour,
            "slot": target_slot or "inner_top",
            "category_confidence": 0.0,
            "colour_confidence": float(pixel_result.get("confidence", 0.0)),
        }
        st.session_state.wardrobe_recognition_result = result
        st.session_state.wardrobe_piece_slot = result["slot"]
        st.session_state.wardrobe_piece_type = ""
        st.session_state.wardrobe_piece_colour = fallback_colour
        return result


def save_analysis_to_wardrobe(result, outfit=None):
    """Save the uploaded piece alone or together with one recommendation."""

    selected_image = st.session_state.get("selected_image")
    if selected_image is None:
        st.error("The selected garment image is no longer available.")
        return
    input_slot = (
        (outfit or {}).get("input_slot")
        or CATEGORY_TO_SLOT.get(result.get("recommendation_category"), "inner_top")
    )
    image_path = save_piece_image(selected_image)
    input_piece = new_piece(
        input_slot,
        result.get("category") or result.get("recommendation_category"),
        result.get("colour"),
        image_path=image_path,
        source="user",
    )
    slots = {slot: None for slot in WARDROBE_SLOTS}
    slots[input_slot] = input_piece
    if outfit is not None:
        for item in outfit.get("items", []):
            item_type = broad_garment_category(
                item["slot"], item.get("type", "")
            )
            if "no outer" in item_type.casefold() or "too hot" in item_type.casefold():
                continue
            piece = new_piece(
                item["slot"],
                item_type,
                item.get("colour"),
                source="recommendation",
            )
            piece["catalogue_image_path"] = str(item.get("image_path", ""))
            slots[item["slot"]] = piece
    saved = new_outfit(
        slots,
        mood=st.session_state.theme,
        style=(outfit or {}).get("style", "To finish later"),
        title="Cove recommendation" if outfit is not None else "Start with this piece",
    )
    st.session_state.wardrobe_store["pieces"].append(input_piece)
    st.session_state.wardrobe_store["outfits"].append(saved)
    persist_wardrobe()
    st.toast("Saved to My wardrobe ❤️")


def wardrobe_piece_image_path(piece):
    """Return a real image only for a piece the user actually saved."""

    candidate = str(piece.get("image_path", ""))
    if candidate and Path(candidate).is_file():
        return candidate
    return ""


def render_wardrobe_piece_preview(piece):
    image_path = wardrobe_piece_image_path(piece)
    if image_path:
        st.image(image_path, use_container_width=True)
    else:
        st.markdown(
            '<div class="wardrobe-icon-preview">'
            f'{garment_icon_svg(piece["slot"], piece.get("colour", ""), piece.get("type", ""))}'
            '</div>',
            unsafe_allow_html=True,
        )


def render_outfit_thumbnail(outfit, slot, *, large=False, surface="card"):
    """Render a directly editable outfit cell for cards and the detail view."""

    piece = outfit.get("slots", {}).get(slot)
    image_path = wardrobe_piece_image_path(piece or {})
    state_class = "photo" if image_path else "icon" if piece else "empty"
    size_class = " large" if large else ""
    key_root = f'{surface}_{outfit["id"]}_{slot}'
    with st.container(border=True):
        st.markdown(
            f'<div class="wardrobe-outfit-preview-marker {state_class}{size_class}"></div>'
            f'<div class="wardrobe-outfit-preview-label">{WARDROBE_SLOT_LABELS[slot]}</div>',
            unsafe_allow_html=True,
        )
        if image_path:
            st.image(image_path, use_container_width=True)
        elif piece:
            st.markdown(
                '<div class="wardrobe-outfit-preview-icon">'
                f'{garment_icon_svg(slot, piece.get("colour", ""), piece.get("type", ""))}'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="wardrobe-outfit-preview-empty">'
                f'{wooden_hanger_svg()}</div>',
                unsafe_allow_html=True,
            )

        if piece is None:
            if st.button(
                "+ Add",
                key=f'{key_root}_add',
                help=f"Add {WARDROBE_SLOT_LABELS[slot].lower()}",
                use_container_width=True,
            ):
                open_wardrobe_capture(outfit["id"], slot)
                st.rerun()
            return

        if not image_path:
            if st.button(
                "Add photo",
                key=f'{key_root}_photo',
                help="Replace the icon with one of your real garments",
                use_container_width=True,
            ):
                open_wardrobe_capture(outfit["id"], slot)
                st.rerun()

        with st.popover("Remove", use_container_width=True):
            st.caption(
                f'Remove {piece.get("type", WARDROBE_SLOT_LABELS[slot])} from this outfit?'
            )
            if st.button(
                "Confirm remove",
                key=f'{key_root}_confirm_remove',
                type="primary",
                use_container_width=True,
            ):
                remove_outfit_slot(
                    st.session_state.wardrobe_store, outfit["id"], slot
                )
                persist_wardrobe()
                st.rerun()


def render_wardrobe_slot(outfit, slot):
    piece = outfit.get("slots", {}).get(slot)
    with st.container(border=True):
        marker_class = " wardrobe-empty-marker" if piece is None else ""
        st.markdown(
            f'<div class="wardrobe-slot-marker{marker_class}"></div>'
            f'<div class="wardrobe-slot-label">{WARDROBE_SLOT_LABELS[slot]}</div>',
            unsafe_allow_html=True,
        )
        if piece is None:
            st.markdown(
                '<div class="wardrobe-empty-hanger">'
                f'{wooden_hanger_svg()}</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "+",
                key=f'wardrobe_add_{outfit["id"]}_{slot}',
                help=f"Add {WARDROBE_SLOT_LABELS[slot].lower()}",
            ):
                open_wardrobe_capture(outfit["id"], slot)
                st.rerun()
            st.markdown(
                '<div class="wardrobe-empty-copy">Add your own</div>',
                unsafe_allow_html=True,
            )
            return

        render_wardrobe_piece_preview(piece)
        st.markdown(
            f'<div class="wardrobe-piece-name">{html.escape(piece.get("type", "Saved piece"))}</div>',
            unsafe_allow_html=True,
        )
        replace_column, remove_column = st.columns(2)
        with replace_column:
            if st.button(
                "Replace",
                key=f'wardrobe_replace_{outfit["id"]}_{slot}',
                use_container_width=True,
            ):
                open_wardrobe_capture(outfit["id"], slot)
                st.rerun()
        with remove_column:
            if st.button(
                "Remove",
                key=f'wardrobe_remove_{outfit["id"]}_{slot}',
                use_container_width=True,
            ):
                remove_outfit_slot(st.session_state.wardrobe_store, outfit["id"], slot)
                persist_wardrobe()
                st.rerun()


def render_saved_outfit(outfit):
    mood_slug = str(outfit.get("mood", "Green")).casefold().replace(" ", "-")
    with st.container(border=True):
        st.markdown(
            f'<div class="wardrobe-outfit-marker mood-{mood_slug}"></div>',
            unsafe_allow_html=True,
        )
        title_column, open_column, action_column = st.columns([0.58, 0.19, 0.23])
        with title_column:
            st.subheader(outfit.get("title", "Saved look"))
            st.markdown(
                f'<span class="wardrobe-group-chip">{html.escape(outfit.get("group", "Everyday"))}</span>'
                f'<span class="wardrobe-card-meta">{html.escape(outfit.get("style", "Personal"))} · '
                f'{html.escape(outfit.get("mood", "Green"))}</span>',
                unsafe_allow_html=True,
            )
        with open_column:
            if st.button(
                "Open",
                key=f'open_outfit_{outfit["id"]}',
                use_container_width=True,
            ):
                st.session_state.wardrobe_selected_outfit_id = outfit["id"]
                st.session_state.app_page = "wardrobe_outfit"
                st.rerun()
        with action_column:
            with st.popover("Edit", use_container_width=True):
                edited_title = st.text_input(
                    "Outfit name",
                    value=outfit.get("title", "Saved look"),
                    key=f'outfit_title_{outfit["id"]}',
                )
                groups = st.session_state.wardrobe_store.get(
                    "outfit_groups", list(DEFAULT_OUTFIT_GROUPS)
                )
                current_group = outfit.get("group", "Everyday")
                edited_group = st.selectbox(
                    "Group",
                    groups,
                    index=groups.index(current_group) if current_group in groups else 0,
                    key=f'outfit_group_{outfit["id"]}',
                )
                if st.button(
                    "Save details",
                    key=f'save_outfit_details_{outfit["id"]}',
                    type="primary",
                    use_container_width=True,
                ):
                    update_outfit_details(
                        st.session_state.wardrobe_store,
                        outfit["id"],
                        title=edited_title,
                        group=edited_group,
                    )
                    persist_wardrobe()
                    st.rerun()
                confirm_delete = st.checkbox(
                    "Confirm whole-outfit deletion",
                    key=f'confirm_delete_outfit_{outfit["id"]}',
                )
                if st.button(
                    "Delete outfit",
                    key=f'delete_outfit_{outfit["id"]}',
                    use_container_width=True,
                    disabled=not confirm_delete,
                ):
                    remove_outfit(st.session_state.wardrobe_store, outfit["id"])
                    persist_wardrobe()
                    st.rerun()
        columns = st.columns(4, gap="small")
        for column, slot in zip(columns, WARDROBE_SLOTS):
            with column:
                render_outfit_thumbnail(outfit, slot)


def render_wardrobe_page():
    weather = st.session_state.weather_data or ensure_current_weather()
    if weather:
        condition = str(weather.get("condition", "Unknown"))
        weather_icons = {
            "Clear": "☀",
            "Cloudy": "☁",
            "Fog": "≋",
            "Rain": "☂",
            "Snow": "❄",
            "Thunderstorm": "ϟ",
        }
        weather_icon = weather_icons.get(condition, "◌")
        temperature = float(weather.get("temperature", weather.get("feels_like", 20)))
        weather_summary = (
            '<div class="wardrobe-heading-weather">'
            f'<span>{weather_icon}</span><div><strong>{temperature:.0f}° · {html.escape(condition)}</strong>'
            f'<small>{html.escape(str(weather.get("city", st.session_state.city)))}</small></div></div>'
        )
    else:
        weather_summary = (
            '<div class="wardrobe-heading-weather"><span>◌</span>'
            '<div><strong>Weather unavailable</strong><small>Using your wardrobe only</small></div></div>'
        )
    navigation, heading, heading_action = st.columns(
        [0.14, 0.72, 0.14], vertical_alignment="center"
    )
    with navigation:
        if st.button("← Home", key="wardrobe_home", use_container_width=True):
            st.session_state.app_page = "home"
            st.rerun()
    with heading:
        st.markdown(
            '<div class="wardrobe-page-heading">'
            f'{wooden_hanger_svg()}'
            '<div><h1>My wardrobe</h1></div>'
            f'{weather_summary}'
            '</div>',
            unsafe_allow_html=True,
        )
    with heading_action:
        if st.button(
            "+ Add",
            key="wardrobe_add_piece",
            type="primary",
            use_container_width=False,
        ):
            open_wardrobe_capture()
            st.rerun()

    outfits_tab, pieces_tab = st.tabs(["My outfits", "My pieces"])
    with outfits_tab:
        groups = st.session_state.wardrobe_store.get(
            "outfit_groups", list(DEFAULT_OUTFIT_GROUPS)
        )
        filter_options = ["All", *groups]
        if st.session_state.wardrobe_group_filter not in filter_options:
            st.session_state.wardrobe_group_filter = "All"
        action_column, filter_column, manage_column, _ = st.columns(
            [0.16, 0.24, 0.14, 0.46], vertical_alignment="bottom"
        )
        with action_column:
            if st.button(
                "+ Empty look",
                key="new_empty_outfit",
                use_container_width=True,
            ):
                st.session_state.wardrobe_store["outfits"].insert(
                    0,
                    new_outfit(
                        mood=st.session_state.theme,
                        title="Untitled look",
                        group=(
                            st.session_state.wardrobe_group_filter
                            if st.session_state.wardrobe_group_filter != "All"
                            else "Everyday"
                        ),
                    ),
                )
                persist_wardrobe()
                st.rerun()
        with filter_column:
            st.selectbox(
                "Show group",
                filter_options,
                key="wardrobe_group_filter",
            )
        with manage_column:
            with st.popover("Manage", use_container_width=True):
                st.markdown(
                    '<div class="outfit-manage-popover-marker"></div>',
                    unsafe_allow_html=True,
                )
                st.caption("Groups")
                new_group_name = st.text_input(
                    "New group",
                    placeholder="e.g. Summer trip",
                    key="new_outfit_group_name",
                )
                if st.button(
                    "Create group",
                    key="create_outfit_group",
                    type="primary",
                    use_container_width=True,
                    disabled=not new_group_name.strip(),
                ):
                    add_outfit_group(
                        st.session_state.wardrobe_store, new_group_name
                    )
                    persist_wardrobe()
                    st.rerun()
                custom_groups = [
                    group for group in groups if group not in DEFAULT_OUTFIT_GROUPS
                ]
                if custom_groups:
                    st.caption("Custom groups")
                    for group in custom_groups:
                        group_name_column, remove_column = st.columns([0.74, 0.26])
                        with group_name_column:
                            st.write(group)
                        with remove_column:
                            if st.button(
                                "×",
                                key=f'delete_group_{group}',
                                help=f"Delete {group}",
                            ):
                                remove_outfit_group(
                                    st.session_state.wardrobe_store, group
                                )
                                persist_wardrobe()
                                st.rerun()
                st.divider()
                with st.expander("Clear outfits"):
                    st.caption("Individual pieces stay in your wardrobe.")
                    confirm_clear_outfits = st.checkbox(
                        "I understand this removes every outfit",
                        key="confirm_clear_all_outfits",
                    )
                    if st.button(
                        "Clear all outfits",
                        key="clear_all_outfits",
                        use_container_width=True,
                        disabled=not confirm_clear_outfits,
                    ):
                        st.session_state.wardrobe_store["outfits"] = []
                        persist_wardrobe()
                        st.rerun()
        outfits = st.session_state.wardrobe_store.get("outfits", [])
        selected_group = st.session_state.wardrobe_group_filter
        if selected_group != "All":
            outfits = [
                outfit
                for outfit in outfits
                if outfit.get("group", "Everyday") == selected_group
            ]
        if not outfits:
            st.markdown(
                '<div class="wardrobe-empty-state">No looks here yet. Save a Cove recommendation or begin an empty look.</div>',
                unsafe_allow_html=True,
            )
        for start in range(0, len(outfits), 2):
            outfit_columns = st.columns(2, gap="medium")
            for column, outfit in zip(outfit_columns, outfits[start : start + 2]):
                with column:
                    render_saved_outfit(outfit)

    with pieces_tab:
        pieces_all = st.session_state.wardrobe_store.get("pieces", [])
        piece_filter_options = ["All", *WARDROBE_SLOTS]
        if st.session_state.wardrobe_piece_filter not in piece_filter_options:
            st.session_state.wardrobe_piece_filter = "All"
        filter_column, manage_column, _ = st.columns(
            [0.58, 0.14, 0.28], vertical_alignment="bottom"
        )
        with filter_column:
            st.pills(
                "Filter pieces",
                piece_filter_options,
                key="wardrobe_piece_filter",
                format_func=lambda value: (
                    "All"
                    if value == "All"
                    else WARDROBE_SLOT_LABELS[value]
                ),
                selection_mode="single",
            )
        with manage_column:
            with st.popover("Manage", use_container_width=True):
                with st.expander("Clear pieces"):
                    st.caption("Saved outfits stay, but the piece library is cleared.")
                    confirm_clear_pieces = st.checkbox(
                        "I understand this removes every piece",
                        key="confirm_clear_all_pieces",
                    )
                    if st.button(
                        "Clear all pieces",
                        key="clear_all_pieces",
                        use_container_width=True,
                        disabled=not confirm_clear_pieces,
                    ):
                        st.session_state.wardrobe_store["pieces"] = []
                        persist_wardrobe()
                        st.rerun()
        selected_piece_filter = st.session_state.wardrobe_piece_filter or "All"
        pieces = filter_pieces_by_slot(pieces_all, selected_piece_filter)
        if not pieces:
            empty_message = (
                "No pieces in this category yet."
                if pieces_all
                else "Circle a favourite garment to begin your private wardrobe."
            )
            st.markdown(
                f'<div class="wardrobe-empty-state">{empty_message}</div>',
                unsafe_allow_html=True,
            )
        for start in range(0, len(pieces), 4):
            columns = st.columns(4, gap="small")
            for column, piece in zip(columns, pieces[start : start + 4]):
                with column:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="wardrobe-piece-marker"></div>'
                            f'<div class="wardrobe-slot-label">{WARDROBE_SLOT_LABELS[piece["slot"]]}</div>',
                            unsafe_allow_html=True,
                        )
                        render_wardrobe_piece_preview(piece)
                        st.markdown(
                            f'<div class="wardrobe-piece-name">{html.escape(piece.get("type", "Saved piece"))}</div>',
                            unsafe_allow_html=True,
                        )
                        with st.popover("Edit", use_container_width=True):
                            edited_name = st.text_input(
                                "Piece name",
                                value=piece.get("type", "Saved piece"),
                                key=f'piece_name_{piece["id"]}',
                            )
                            if st.button(
                                "Save name",
                                key=f'save_piece_name_{piece["id"]}',
                                type="primary",
                                use_container_width=True,
                            ):
                                try:
                                    update_piece_name(
                                        st.session_state.wardrobe_store,
                                        piece["id"],
                                        edited_name,
                                    )
                                except ValueError as error:
                                    st.error(str(error))
                                else:
                                    persist_wardrobe()
                                    st.toast("Piece renamed")
                                    st.rerun()
                            st.divider()
                            confirm_delete = st.checkbox(
                                "Confirm deletion",
                                key=f'confirm_delete_piece_{piece["id"]}',
                            )
                            if st.button(
                                "Delete piece",
                                key=f'delete_piece_{piece["id"]}',
                                disabled=not confirm_delete,
                                use_container_width=True,
                            ):
                                remove_piece(
                                    st.session_state.wardrobe_store, piece["id"]
                                )
                                persist_wardrobe()
                                st.rerun()

    render_style_discovery("wardrobe")
    if st.session_state.wardrobe_dialog_open:
        render_wardrobe_capture_dialog()


def render_outfit_detail_page():
    """Render one outfit at a comfortable inspection size."""

    outfit = find_outfit(
        st.session_state.wardrobe_store,
        st.session_state.wardrobe_selected_outfit_id,
    )
    if outfit is None:
        st.session_state.app_page = "wardrobe"
        st.rerun()

    back_column, title_column = st.columns(
        [0.16, 0.84], vertical_alignment="center"
    )
    with back_column:
        if st.button("← Wardrobe", use_container_width=True):
            st.session_state.app_page = "wardrobe"
            st.rerun()
    with title_column:
        st.markdown(
            '<div class="wardrobe-detail-heading">'
            f'<span>{html.escape(outfit.get("group", "Everyday"))}</span>'
            f'<h1>{html.escape(outfit.get("title", "Saved look"))}</h1>'
            f'<p>{html.escape(outfit.get("style", "Personal"))} · '
            f'{html.escape(outfit.get("mood", "Green"))} mood</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        st.markdown(
            '<div class="wardrobe-detail-marker"></div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(4, gap="medium")
        for column, slot in zip(columns, WARDROBE_SLOTS):
            with column:
                render_outfit_thumbnail(
                    outfit, slot, large=True, surface="detail"
                )
    render_style_discovery("wardrobe")
    if st.session_state.wardrobe_dialog_open:
        render_wardrobe_capture_dialog()


@st.dialog("Add to your wardrobe", width="large")
def render_wardrobe_capture_dialog():
    target = st.session_state.wardrobe_capture_target or {}
    target_slot = target.get("slot")
    heading = (
        f'Replace {WARDROBE_SLOT_LABELS[target_slot].lower()}'
        if target_slot and find_outfit(
            st.session_state.wardrobe_store, target.get("outfit_id")
        )
        and find_outfit(
            st.session_state.wardrobe_store, target.get("outfit_id")
        ).get("slots", {}).get(target_slot)
        else f'Add {WARDROBE_SLOT_LABELS[target_slot].lower()}'
        if target_slot
        else "Add a piece"
    )
    title_column, close_column = st.columns([0.78, 0.22], vertical_alignment="center")
    with title_column:
        st.markdown(
            '<div class="wardrobe-page-heading">'
            f'{wooden_hanger_svg()}<div><h1>{html.escape(heading)}</h1>'
            '<p>Choose one you saved, or circle it from a photo.</p></div></div>',
            unsafe_allow_html=True,
        )
    with close_column:
        if st.button("Close", use_container_width=False, key="capture_back"):
            reset_wardrobe_capture()
            st.rerun()

    if target_slot:
        matching = pieces_for_slot(
            st.session_state.wardrobe_store, target_slot
        )
        if matching:
            with st.container(border=True):
                st.caption("Already in your wardrobe")
                preview_column, choice_column = st.columns(
                    [0.22, 0.78], vertical_alignment="center"
                )
                with choice_column:
                    selected_id = st.selectbox(
                        "Search saved pieces",
                        [piece["id"] for piece in matching],
                        format_func=lambda piece_id: next(
                            piece.get("type", "Saved piece")
                            for piece in matching
                            if piece["id"] == piece_id
                        ),
                        key="existing_piece_choice",
                    )
                    selected_piece = next(
                        piece for piece in matching if piece["id"] == selected_id
                    )
                    st.caption(
                        f'{selected_piece.get("colour") or "Colour not set"} · '
                        f'{WARDROBE_SLOT_LABELS[selected_piece["slot"]]}'
                    )
                    if st.button(
                        "Use this piece",
                        key="use_existing_piece",
                        type="primary",
                        use_container_width=False,
                    ):
                        set_outfit_slot(
                            st.session_state.wardrobe_store,
                            target["outfit_id"],
                            target_slot,
                            dict(selected_piece),
                        )
                        persist_wardrobe()
                        return_page = st.session_state.wardrobe_capture_return_page
                        reset_wardrobe_capture()
                        st.session_state.app_page = return_page
                        st.rerun()
                with preview_column:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="wardrobe-picker-marker"></div>',
                            unsafe_allow_html=True,
                        )
                        render_wardrobe_piece_preview(selected_piece)
            st.caption("Or circle a different garment below.")

    if st.session_state.wardrobe_uploaded_bytes is None:
        uploaded = st.file_uploader(
            "Choose a photo",
            type=["jpg", "jpeg", "png"],
            key=f'wardrobe_uploader_{st.session_state.wardrobe_uploader_version}',
            help="A clear flat-lay or a well-lit photo gives the cleanest wardrobe image.",
        )
        if uploaded is not None:
            st.session_state.wardrobe_uploaded_bytes = uploaded.getvalue()
            st.session_state.wardrobe_uploaded_name = uploaded.name
            st.rerun()
        st.markdown(
            '<div class="wardrobe-empty-state">Choose a photo, then draw gently around the garment.</div>',
            unsafe_allow_html=True,
        )
        return

    original = Image.open(io.BytesIO(st.session_state.wardrobe_uploaded_bytes)).convert("RGB")
    canvas_column, detail_column = st.columns([1.08, 0.92], gap="large")
    with canvas_column:
        if not st.session_state.wardrobe_selection_ready:
            canvas_frame, frame_information = create_fixed_frame(original)
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
                    f'wardrobe_canvas_{st.session_state.wardrobe_uploaded_name}_'
                    f'{st.session_state.wardrobe_canvas_version}'
                ),
            )
            objects = (
                canvas_result.json_data.get("objects", [])
                if canvas_result.json_data is not None
                else []
            )
            if objects:
                selected = crop_from_drawing(original, objects[-1], frame_information)
                if selected is not None:
                    st.session_state.wardrobe_selected_image = selected
                    st.session_state.wardrobe_selection_ready = True
                    st.rerun()
            if st.button(
                "Use the whole photo",
                key="wardrobe_use_whole_photo",
                use_container_width=True,
            ):
                st.session_state.wardrobe_selected_image = original
                st.session_state.wardrobe_selection_ready = True
                st.rerun()
        else:
            st.image(st.session_state.wardrobe_selected_image, use_container_width=True)
            if st.button("Draw again", key="wardrobe_draw_again", use_container_width=True):
                st.session_state.wardrobe_selected_image = None
                st.session_state.wardrobe_selection_ready = False
                st.session_state.wardrobe_recognition_result = None
                st.session_state.wardrobe_recognition_error = None
                for widget_key in (
                    "wardrobe_piece_slot",
                    "wardrobe_piece_type",
                    "wardrobe_piece_colour",
                ):
                    st.session_state.pop(widget_key, None)
                st.session_state.wardrobe_canvas_version += 1
                st.rerun()

    with detail_column:
        recognition = None
        if st.session_state.wardrobe_selection_ready:
            recognition = recognise_wardrobe_piece(
                st.session_state.wardrobe_selected_image,
                original,
                target_slot,
            )
            if recognition.get("category") or recognition.get("colour"):
                suggestions = " · ".join(
                    value
                    for value in (
                        recognition.get("category"),
                        recognition.get("colour"),
                    )
                    if value
                )
                st.success(f"Recognised: {suggestions}")
                st.caption("These are suggestions — erase or change either field below.")
            elif st.session_state.wardrobe_recognition_error:
                st.caption("Automatic recognition was unavailable; you can enter the details manually.")

        st.caption("A lightweight cleanup removes simple plain backgrounds when you save.")
        slot_index = WARDROBE_SLOTS.index(target_slot) if target_slot else 0
        slot = st.selectbox(
            "Garment place",
            WARDROBE_SLOTS,
            index=slot_index,
            format_func=lambda value: WARDROBE_SLOT_LABELS[value],
            disabled=bool(target_slot),
            key="wardrobe_piece_slot",
        )
        item_type = st.text_input(
            "What is it?",
            value="",
            placeholder="e.g. Jacket",
            key="wardrobe_piece_type",
        )
        colour = st.selectbox(
            "Main colour",
            [""] + list(ICON_COLOURS),
            format_func=lambda value: value or "Not set",
            key="wardrobe_piece_colour",
        )
        if st.session_state.wardrobe_selection_ready and st.button(
            "Save to my wardrobe ❤️",
            key="save_wardrobe_piece",
            type="primary",
            use_container_width=True,
        ):
            image_path = save_piece_image(st.session_state.wardrobe_selected_image)
            piece = new_piece(slot, item_type, colour, image_path=image_path)
            st.session_state.wardrobe_store["pieces"].append(piece)
            if target.get("outfit_id") and target_slot:
                set_outfit_slot(
                    st.session_state.wardrobe_store,
                    target["outfit_id"],
                    target_slot,
                    dict(piece),
                )
            persist_wardrobe()
            return_page = st.session_state.wardrobe_capture_return_page
            reset_wardrobe_capture()
            st.session_state.app_page = return_page
            st.rerun()


initialise_state()
render_theme_css()

if st.session_state.app_page != "home":
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(ellipse at 7% 0%,color-mix(in srgb,var(--fashion-primary) 13%,transparent),transparent 34rem),
                    radial-gradient(ellipse at 96% 10%,rgba(226,187,159,.16),transparent 31rem),
                    linear-gradient(145deg,#fcfaf5 0%,#f8f3eb 56%,#f5eee5 100%) !important;
            }

            /* Floating surfaces use the same warm porcelain as the page. */
            body [data-testid="stDialog"] [role="dialog"],
            body [data-testid="stPopoverBody"],
            body [data-baseweb="popover"] > div {
                color:var(--fashion-ink) !important;
                border:1px solid color-mix(in srgb,var(--fashion-primary) 18%,#ded3c6) !important;
                background:
                    radial-gradient(ellipse at 7% 0%,color-mix(in srgb,var(--fashion-primary) 13%,transparent),transparent 34rem),
                    radial-gradient(ellipse at 96% 10%,rgba(226,187,159,.16),transparent 31rem),
                    linear-gradient(145deg,#fcfaf5 0%,#f8f3eb 56%,#f5eee5 100%) !important;
                box-shadow:
                    inset 0 1px 0 rgba(255,255,255,.78),
                    0 22px 58px rgba(77,59,43,.15) !important;
                backdrop-filter:blur(22px) saturate(1.03) !important;
                -webkit-backdrop-filter:blur(22px) saturate(1.03) !important;
            }

            body [data-testid="stDialog"] [role="dialog"] {
                border-radius:28px 28px 16px 28px !important;
                overflow:hidden;
            }

            body [data-testid="stPopoverBody"],
            body [data-baseweb="popover"] > div {
                border-radius:20px 20px 11px 20px !important;
                overflow:hidden;
            }

            body [data-testid="stDialog"] {
                background:color-mix(in srgb,var(--fashion-primary) 8%,rgba(74,61,50,.18)) !important;
                backdrop-filter:blur(4px) saturate(.94) !important;
                -webkit-backdrop-filter:blur(4px) saturate(.94) !important;
            }

            body [data-testid="stDialog"] [data-testid="stFileUploaderDropzone"] {
                border:1px dashed color-mix(in srgb,var(--fashion-primary) 24%,#d8cec1) !important;
                border-radius:18px 18px 10px 18px !important;
                background:color-mix(in srgb,var(--fashion-primary) 5%,#f5f0e9) !important;
            }

            body [data-testid="stDialog"] [data-baseweb="select"] > div,
            body [data-testid="stDialog"] input,
            body [data-testid="stDialog"] textarea,
            body [data-testid="stPopoverBody"] [data-baseweb="select"] > div,
            body [data-testid="stPopoverBody"] input,
            body [data-testid="stPopoverBody"] textarea,
            body [data-baseweb="popover"] [data-baseweb="select"] > div,
            body [data-baseweb="popover"] input,
            body [data-baseweb="popover"] textarea {
                color:var(--fashion-ink) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 20%,#d9d0c4) !important;
                background:color-mix(in srgb,var(--fashion-primary) 5%,#f5f0e9) !important;
                box-shadow:inset 0 1px 2px rgba(83,62,44,.05) !important;
            }

            body [data-testid="stDialog"] h1,
            body [data-testid="stDialog"] h2,
            body [data-testid="stDialog"] h3,
            body [data-testid="stPopoverBody"] h1,
            body [data-testid="stPopoverBody"] h2,
            body [data-testid="stPopoverBody"] h3,
            body [data-baseweb="popover"] h1,
            body [data-baseweb="popover"] h2,
            body [data-baseweb="popover"] h3 {
                color:var(--fashion-ink) !important;
            }

            body .stApp [data-testid="stMainBlockContainer"] .stButton > button,
            body .stApp [data-testid="stMainBlockContainer"] .stFormSubmitButton > button,
            body .stApp [data-testid="stMainBlockContainer"] [data-testid="stPopover"] > button,
            body .stApp [data-testid="stMainBlockContainer"] button[data-testid="stPopoverButton"] {
                color: var(--fashion-strong) !important;
                border:1px solid color-mix(in srgb,var(--fashion-primary) 30%,#d8cfc3) !important;
                background:color-mix(in srgb,var(--fashion-primary) 13%,#f6efe6) !important;
                box-shadow:
                    inset 0 1px 0 rgba(255,255,255,.66),
                    inset 0 -1px 0 rgba(104,78,53,.05),
                    0 5px 13px rgba(76,61,47,.08) !important;
                filter:none !important;
                transform:none !important;
                transition:background .18s ease,border-color .18s ease,box-shadow .18s ease,color .18s ease !important;
            }

            body .stApp [data-testid="stMainBlockContainer"] .stButton > button[kind="primary"],
            body .stApp [data-testid="stMainBlockContainer"] .stFormSubmitButton > button[kind="primary"] {
                color: var(--fashion-strong) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 45%,#c9bbaa) !important;
                background:color-mix(in srgb,var(--fashion-primary) 23%,#f2e9de) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.62),0 6px 15px color-mix(in srgb,var(--fashion-primary) 11%,rgba(76,61,47,.09)) !important;
            }

            body .stApp [data-testid="stMainBlockContainer"] .stButton > button:hover,
            body .stApp [data-testid="stMainBlockContainer"] .stFormSubmitButton > button:hover,
            body .stApp [data-testid="stMainBlockContainer"] [data-testid="stPopover"] > button:hover,
            body .stApp [data-testid="stMainBlockContainer"] button[data-testid="stPopoverButton"]:hover {
                color: var(--fashion-strong) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 58%,#c2b29f) !important;
                background:color-mix(in srgb,var(--fashion-primary) 29%,#efe5d9) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.58),0 7px 17px color-mix(in srgb,var(--fashion-primary) 15%,rgba(76,61,47,.10)) !important;
                filter:none !important;
                transform:none !important;
            }

            body .stApp [data-testid="stMainBlockContainer"] div[class*="st-key-delete_outfit_"] button:not(:disabled),
            body .stApp [data-testid="stMainBlockContainer"] div[class*="st-key-clear_all_outfits"] button:not(:disabled),
            body .stApp [data-testid="stMainBlockContainer"] div[class*="confirm_remove"] button:not(:disabled) {
                color: #8e4a42 !important;
                border-color: rgba(181,99,84,.25) !important;
                background:#f6e7e1 !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.58),0 5px 13px rgba(145,60,60,.07) !important;
            }

            body .stApp div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-marker),
            body .stApp div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-detail-marker),
            body .stApp div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-piece-marker),
            body .stApp div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-picker-marker) {
                border-color:rgba(222,210,195,.78) !important;
                background:rgba(250,247,241,.76) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.72),0 12px 30px rgba(78,62,47,.07) !important;
                backdrop-filter:blur(14px) saturate(1.04) !important;
                -webkit-backdrop-filter:blur(14px) saturate(1.04) !important;
            }

            body .stApp div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .wardrobe-outfit-preview-marker) {
                border-color:color-mix(in srgb,var(--fashion-primary) 15%,#ded5c9) !important;
                background:color-mix(in srgb,var(--fashion-primary) 5%,#f7f2ea) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.70),0 5px 14px rgba(78,62,47,.06) !important;
            }

            body .stApp .style-discovery {
                border-color:rgba(222,210,195,.74) !important;
                background:rgba(249,245,238,.78) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.70),0 10px 26px rgba(78,62,47,.06) !important;
            }

            body .stApp [data-baseweb="select"] > div {
                border-color:color-mix(in srgb,var(--fashion-primary) 18%,#d9d0c4) !important;
                background:color-mix(in srgb,var(--fashion-primary) 5%,#f5f0e9) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.58) !important;
            }

            body .stApp [data-baseweb="tab"][aria-selected="true"] {
                color:var(--fashion-strong) !important;
                border-bottom-color:var(--fashion-primary) !important;
            }
        </style>
        <div class="app-wordmark">
            <span class="app-brand-lockup">
                <span class="app-brand-name">Cove</span>
                <span class="app-brand-tagline">Your wardrobe</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )



if st.session_state.app_page == "wardrobe":
    render_wardrobe_page()
    st.stop()


if st.session_state.app_page == "wardrobe_outfit":
    render_outfit_detail_page()
    st.stop()


if st.session_state.app_page == "wardrobe_capture":
    # Migrate sessions left open on the former full-page capture route.
    st.session_state.app_page = "wardrobe"
    st.session_state.wardrobe_dialog_open = True
    st.rerun()


# =========================================================
# Opening mode-selection page
# =========================================================

if st.session_state.app_page == "home":
    st.markdown(
        """
        <style>
            .stButton > button {
                min-height: 56px;
                padding: .95rem 1.4rem;
                border: 1px solid var(--fashion-strong) !important;
                border-radius: 999px;
                background: var(--fashion-strong) !important;
                color: white !important;
                box-shadow: 0 13px 30px color-mix(in srgb, var(--fashion-strong) 24%, transparent) !important;
                filter: none !important;
                font-size: .82rem;
                letter-spacing: .12em;
                line-height: 1.2;
                text-align: center;
                white-space: pre-wrap;
                transition:
                    transform 0.2s ease,
                    background-color 0.2s ease,
                    border-color 0.2s ease,
                    color 0.2s ease,
                    box-shadow 0.2s ease;
            }

            .stButton > button:hover {
                transform: translateY(-2px) !important;
                border-color: var(--fashion-strong) !important;
                color: white !important;
                background: var(--fashion-strong) !important;
                filter: brightness(.94) !important;
                box-shadow:
                    0 18px 38px
                    color-mix(in srgb, var(--fashion-primary) 22%, transparent) !important;
            }

            .stButton > button[kind="primary"] {
                border-color: var(--fashion-strong) !important;
                background: var(--fashion-strong) !important;
                color: white !important;
                box-shadow: 0 13px 30px color-mix(in srgb, var(--fashion-strong) 24%, transparent) !important;
            }

            .stButton > button[kind="primary"]:hover {
                border-color: var(--fashion-strong) !important;
                background: var(--fashion-strong) !important;
                color: white !important;
                filter: brightness(.94) !important;
                box-shadow: 0 18px 38px color-mix(in srgb, var(--fashion-primary) 22%, transparent) !important;
                transform: translateY(-2px) !important;
            }

            /* Weather controls belong to the warm dock, not Streamlit's white defaults. */
            div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"],
            div[data-testid="stForm"] div[data-testid="stTextInput"] input,
            div[data-testid="stForm"] .stFormSubmitButton > button {
                height: 40px !important;
                min-height: 40px !important;
                max-height: 40px !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stForm"] .stFormSubmitButton > button {
                padding: 0 .9rem !important;
                border: 1px solid color-mix(in srgb,var(--fashion-primary) 22%,#e3c9a6) !important;
                background:color-mix(in srgb,var(--fashion-primary) 18%,var(--fashion-soft)) !important;
                color: var(--fashion-strong) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.48),0 6px 15px color-mix(in srgb,var(--fashion-primary) 12%,rgba(93,67,43,.08)) !important;
                font-size:.84rem !important;
                letter-spacing:.025em !important;
            }

            div[data-testid="stForm"] .stFormSubmitButton > button:hover {
                transform:translateY(-1px) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 48%,#dbb783) !important;
                background:color-mix(in srgb,var(--fashion-primary) 30%,var(--fashion-soft)) !important;
                color:var(--fashion-strong) !important;
                filter:none !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.42),0 9px 19px color-mix(in srgb,var(--fashion-primary) 17%,rgba(93,67,43,.10)) !important;
            }

            div[class*="st-key-set_city_button"] button,
            div[class*="st-key-favourite_city_button"] button {
                height:40px !important;
                min-height:40px !important;
                max-height:40px !important;
                padding:0 !important;
                background:color-mix(in srgb,var(--fashion-primary) 18%,var(--fashion-soft)) !important;
                border-color:color-mix(in srgb,var(--fashion-primary) 22%,#e3c9a6) !important;
            }

            body .stApp div[class*="st-key-set_city_button"],
            body .stApp div[class*="st-key-favourite_city_button"],
            body .stApp div[class*="st-key-set_city_button"] .stFormSubmitButton,
            body .stApp div[class*="st-key-favourite_city_button"] .stFormSubmitButton {
                height:40px !important;
                min-height:40px !important;
                max-height:40px !important;
            }

            body .stApp div[class*="st-key-set_city_button"] button,
            body .stApp div[class*="st-key-favourite_city_button"] button {
                height:40px !important;
                min-height:40px !important;
                max-height:40px !important;
                padding:0 .55rem !important;
                box-sizing:border-box !important;
            }

            div[class*="st-key-favourite_city_button"] button {
                color:#c64f5c !important;
                font-size:1.12rem !important;
            }

            .weather-favourite-mark { color:#c64f5c !important; }

            .saved-cities-label {
                position:relative;
                z-index:2;
                margin:.16rem 0 .32rem;
                color:var(--fashion-ink);
                font-size:.78rem;
                line-height:1.35;
            }

            .saved-cities-label span {
                color:var(--fashion-primary);
                font-size:.92rem;
            }

            div[class*="st-key-saved_city_choice"] {
                margin-top:0 !important;
                overflow:visible !important;
            }

            div[class*="st-key-saved_city_choice"] [data-testid="stPills"],
            div[class*="st-key-saved_city_choice"] [data-testid="stButtonGroup"] {
                margin-top:0 !important;
                overflow:visible !important;
            }

            div[class*="st-key-saved_city_choice"] [data-testid="stPills"] button,
            div[class*="st-key-saved_city_choice"] [data-testid="stButtonGroup"] button {
                min-height:40px !important;
                padding:.42rem .75rem !important;
                border:1px solid color-mix(in srgb,var(--fashion-primary) 26%,#c7c3bb) !important;
                border-radius:999px !important;
                background:color-mix(in srgb,var(--fashion-soft) 66%,rgba(255,255,255,.72)) !important;
                color:var(--fashion-ink) !important;
                box-shadow:inset 0 1px 0 rgba(255,255,255,.68),0 4px 10px rgba(72,66,58,.06) !important;
                letter-spacing:.01em !important;
            }

            div[class*="st-key-saved_city_choice"] [data-testid="stPills"] button[aria-checked="true"],
            div[class*="st-key-saved_city_choice"] [data-testid="stButtonGroup"] button[aria-checked="true"] {
                border-color:color-mix(in srgb,var(--fashion-primary) 58%,#aaa69e) !important;
                background:color-mix(in srgb,var(--fashion-primary) 20%,var(--fashion-soft)) !important;
                color:var(--fashion-strong) !important;
                box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--fashion-primary) 16%,transparent),0 5px 12px color-mix(in srgb,var(--fashion-primary) 10%,transparent) !important;
            }

            div[data-testid="stRadio"] [role="radiogroup"] {
                display: flex;
                align-items: center;
                gap: .72rem;
                flex-wrap: nowrap;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"] {
                position: relative;
                width: 48px;
                min-width: 48px;
                height: 36px;
                min-height: 36px;
                padding: 0 !important;
                overflow: visible;
                border: 2px solid rgba(255,255,255,.72);
                border-radius: 38% 24% 42% 28% / 32% 44% 26% 40%;
                box-shadow: 0 7px 15px rgba(45,55,50,.12), inset 0 0 0 1px rgba(35,45,40,.06);
                transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-child(1) { background: #78a6cd; }
            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-child(2) { background: #78a985; }
            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-child(3) { background: #dea0b4; }
            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-child(4) { background: #c9ad87; }
            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:nth-child(5) {
                background: #fbfbf8;
                border-color: #c9cbc7;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div {
                width: 100%;
                height: 100%;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"]
            > div > div > div:first-child {
                display: none;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"] p {
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                clip-path: inset(50%);
                white-space: nowrap;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"]::after {
                content: "✦";
                position: absolute;
                top: -8px;
                right: -6px;
                display: grid;
                width: 18px;
                height: 18px;
                place-items: center;
                border-radius: 7px 10px 7px 9px;
                background: var(--fashion-strong);
                color: white;
                font-size: .62rem;
                opacity: 0;
                transform: scale(.6) rotate(-12deg);
                transition: opacity .18s ease, transform .18s ease;
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {
                transform: translateY(-2px) rotate(-1deg);
                filter: saturate(1.08);
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked) {
                transform: translateY(-2px) rotate(-2deg);
                box-shadow:
                    0 0 0 3px rgba(255,255,255,.9),
                    0 0 0 5px var(--fashion-strong),
                    0 10px 20px rgba(45,55,50,.16);
            }

            div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked)::after {
                opacity: 1;
                transform: scale(1) rotate(7deg);
            }

            .home-wordmark {
                max-width: 1180px;
                margin: .25rem auto 1rem;
            }

            .weather-window-scene {
                min-height: 720px;
                max-width: 1080px;
                margin: .25rem auto 0;
            }

            .weather-neighbourhood {
                right: 3%;
                left: 8%;
                height: 39%;
                transform: scale(1.12);
                transform-origin: bottom center;
            }

            .weather-window-info {
                top: 1.25rem;
                right: 1.4rem;
                bottom: auto;
                left: auto;
                width: 300px;
                min-height: auto;
                padding: .4rem .25rem;
                border: 0;
                border-radius: 0;
                background: transparent;
                box-shadow: none;
                backdrop-filter: none;
            }

            .weather-illustration {
                inset: 1.2rem .4rem 4.2rem;
            }

            .weather-orb {
                top: 14%;
                right: 27%;
                width: 12rem;
            }

            .weather-clear .weather-face {
                top: 28%;
                right: 31.5%;
            }

            .weather-clear .weather-cloud-form {
                top: 48%;
                left: 26%;
                width: 45%;
                transform: scale(.9);
            }

            .home-brand-lockup {
                display: flex;
                flex-direction: column;
                gap: .05rem;
            }

            .home-tagline {
                color: var(--fashion-muted);
                font-size: .72rem;
                font-weight: 560;
                letter-spacing: .035em;
            }

            .home-dock-marker {
                height: 0;
                overflow: hidden;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker) {
                position: relative;
                z-index: 20;
                width: min(430px, calc(100% - 3rem));
                margin: -445px 2.35rem 4.25rem auto;
                padding: 1.15rem 1.2rem 1rem;
                border: 1px solid rgba(255,255,255,.72) !important;
                border-radius: 28px 28px 12px 28px !important;
                --dock-x: 0px;
                --dock-y: 0px;
                background: rgba(255, 252, 247, .54) !important;
                box-shadow: 0 24px 58px rgba(65,61,54,.15) !important;
                backdrop-filter: blur(24px) saturate(1.08);
                transform: translate(var(--dock-x), var(--dock-y));
                cursor: grab;
                touch-action: none;
                user-select: none;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker).is-dragging {
                cursor: grabbing;
                box-shadow: 0 30px 72px rgba(65,61,54,.22) !important;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            :is(input, button, label, [role="radio"], [data-testid="stPills"]) {
                cursor: auto;
                user-select: auto;
                touch-action: auto;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            > div[data-testid="stVerticalBlock"] {
                gap: .55rem;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.city-panel-marker) {
                margin-bottom: .7rem;
                background: rgba(255,255,255,.48);
                box-shadow: none;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            .city-panel-marker {
                margin-bottom: .25rem;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            .home-greeting {
                margin-top: .45rem;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            div[data-testid="stPills"] {
                margin-top: -.15rem;
            }

            div[data-testid="stLayoutWrapper"]:has(.home-dock-marker)
            div[data-testid="stPills"] button {
                min-height: 34px;
                padding: .32rem .7rem;
                letter-spacing: .01em;
            }

            @media (max-width: 760px) {
                .opening-title {
                    font-size: clamp(1.9rem, 9vw, 2.65rem);
                    line-height: 1;
                }
                .home-wordmark { margin-bottom: 1.2rem; }
                .weather-window-scene {
                    min-height: 540px;
                    margin-top: 0;
                }
                div[data-testid="stLayoutWrapper"]:has(.home-dock-marker) {
                    width: calc(100% - 1.2rem);
                    margin: -76px auto 2rem;
                    padding: 1.3rem 1rem 1rem;
                }
                .weather-illustration { inset: 3rem .65rem 6rem; }
                .weather-orb { width: 8rem; right: 9%; }
                .weather-clear .weather-face { top: 30%; right: 17.5%; }
                .weather-clear .weather-cloud-form {
                    top: 55%;
                    left: 4%;
                    width: 42%;
                    transform: scale(.84);
                }
                .weather-cloud-form { left: 10%; width: 66%; }
                .weather-window-temperature { font-size: 1.7rem; }
                .weather-neighbourhood { left: 2%; transform: none; }
                .weather-window-info {
                    top: auto;
                    right: 1.4rem;
                    bottom: 1.35rem;
                    left: 1.4rem;
                    width: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.weather_refresh_requested:
        with st.spinner("Opening your saved city..."):
            ensure_current_weather()
        st.session_state.weather_refresh_requested = False

    home_weather = st.session_state.weather_data
    home_weather_error = st.session_state.weather_error

    st.markdown(
        """
        <div class="home-wordmark">
            <span class="home-brand-lockup">
                <span class="home-brand">Cove</span>
                <span class="home-tagline">Your wardrobe</span>
            </span>
            <span class="home-edition">Daily styling</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_weather_window(
        home_weather,
        home_weather_error,
    )

    with st.container(border=False):
        st.markdown(
            '<div class="home-dock-marker"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="city-panel-marker">
                <div class="city-panel-icon" aria-hidden="true">
                    <span class="city-location-pin"></span>
                </div>
                <div>
                    <div class="city-panel-title">Where are you dressing for?</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_is_saved = any(
            saved_city.casefold() == st.session_state.city_input.strip().casefold()
            for saved_city in st.session_state.saved_cities
        )
        with st.form("weather_city_form", border=False):
            city_column, save_city_column, heart_column = st.columns(
                [0.58, 0.27, 0.15],
                vertical_alignment="center",
            )
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
                    "Set",
                    key="set_city_button",
                    use_container_width=True,
                )
            with heart_column:
                favourite_submitted = st.form_submit_button(
                    "♥" if current_is_saved else "♡",
                    key="favourite_city_button",
                    use_container_width=True,
                    help="Save or remove this city from your favourites.",
                )

        entered_city = st.session_state.city_input.strip()
        if city_submitted and entered_city:
            st.session_state.city = entered_city
            st.session_state.weather_data = None
            st.session_state.weather_error = None
            st.session_state.saved_city_choice = None
            with st.spinner("Opening your weather window..."):
                ensure_current_weather()
            st.rerun()

        if favourite_submitted and entered_city:
            updated_cities = toggle_favourite_city(
                st.session_state.saved_cities,
                entered_city,
            )
            try:
                save_favourite_cities(updated_cities)
            except OSError:
                st.error("This city could not be saved locally.")
            else:
                st.session_state.saved_cities = updated_cities
                st.session_state.saved_city_choice = None
                st.rerun()

        if st.session_state.saved_cities:
            st.markdown(
                '<div class="saved-cities-label">Saved cities <span>♥</span></div>',
                unsafe_allow_html=True,
            )
            saved_city_choice = st.pills(
                "Saved cities",
                st.session_state.saved_cities,
                key="saved_city_choice",
                width="stretch",
                on_change=apply_saved_city_choice,
                label_visibility="collapsed",
            )

        if (
            "theme_picker" not in st.session_state
            or st.session_state.theme_picker not in THEMES
        ):
            st.session_state.theme_picker = st.session_state.theme

        st.radio(
            "Choose your mood",
            list(THEMES),
            horizontal=True,
            key="theme_picker",
            on_change=apply_theme_choice,
        )

        start_column, wardrobe_column = st.columns([3, 1], gap="small")
        with start_column:
            if st.button(
                "START WITH A PIECE  →",
                type="primary",
                use_container_width=True,
                key="style_mode_button",
            ):
                select_mode("lifestyle")
                st.rerun()
        with wardrobe_column:
            if st.button(
                "WARDROBE",
                use_container_width=True,
                key="open_wardrobe_button",
            ):
                st.session_state.app_page = "wardrobe"
                st.session_state.image_mode = None
                st.rerun()

    enable_home_dock_dragging()

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
    if st.session_state.selection_ready:
        render_style_discovery("recommendation")
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
