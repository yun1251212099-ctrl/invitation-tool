#!/usr/bin/env python3
"""批量邀请函生成工具 — Streamlit Web 应用 v3 (Cloud)"""

import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pilmoji import Pilmoji
from psd_tools import PSDImage

APP_DIR = Path(__file__).parent
FONTS_DIR = APP_DIR / "fonts"

st.set_page_config(page_title="批量邀请函生成", page_icon="📨", layout="wide")
st.markdown(
    """
    <div class="apple-hero">
      <h1>批量邀请函生成工具</h1>
      <p>上传模板与名单，预览确认后一键批量生成并下载压缩包。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Streamlit 内建上传组件默认是英文，这里统一覆盖成简体中文提示，并优化界面层级。
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    .apple-hero {
        padding: 1.1rem 0 0.6rem 0;
        border-bottom: 1px solid rgba(120, 120, 128, 0.25);
        margin-bottom: 1rem;
    }
    .apple-hero h1 {
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .apple-hero p {
        margin-top: 0.55rem;
        margin-bottom: 0.25rem;
        font-size: 1.05rem;
        color: rgba(128, 128, 132, 0.95);
    }
    .apple-section-title {
        margin: 1.1rem 0 0.35rem 0;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .apple-info-card {
        border: 1px solid rgba(120, 120, 128, 0.3);
        border-radius: 14px;
        padding: 0.8rem 0.95rem;
        margin: 0.2rem 0 0.8rem 0;
        background: rgba(120, 120, 128, 0.08);
    }
    .apple-info-card strong {
        display: block;
        margin-bottom: 0.25rem;
        font-size: 1rem;
    }
    .apple-info-card span {
        font-size: 0.92rem;
        color: rgba(128, 128, 132, 0.95);
    }
    .action-card {
        border: 1px solid rgba(120, 120, 128, 0.35);
        border-radius: 12px;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.55rem;
        background: rgba(120, 120, 128, 0.06);
    }
    .action-card strong {
        display: block;
        font-size: 0.98rem;
        margin-bottom: 0.2rem;
    }
    .action-card span {
        display: block;
        font-size: 0.86rem;
        color: rgba(128, 128, 132, 0.95);
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 1px solid rgba(120, 120, 128, 0.35);
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
    [data-testid="stFileUploader"] small {
        font-size: 0.9rem;
        color: rgba(128, 128, 132, 0.92);
    }
    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 10px;
    }
    [data-testid="stButton"] > button {
        border-radius: 12px;
        min-height: 2.7rem;
    }
    [data-testid="stDownloadButton"] > button {
        border-radius: 12px;
        min-height: 2.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── password gate ────────────────────────────────────────
import hashlib
_PASS = "950621"
_TOKEN = hashlib.sha256(_PASS.encode()).hexdigest()[:16]

_authenticated = False
if st.query_params.get("token") == _TOKEN:
    _authenticated = True
elif st.session_state.get("authenticated"):
    _authenticated = True

if not _authenticated:
    pwd = st.text_input("\u8bf7\u8f93\u5165\u8bbf\u95ee\u5bc6\u7801", type="password")
    if pwd == _PASS:
        st.session_state.authenticated = True
        st.query_params["token"] = _TOKEN
        st.rerun()
    elif pwd:
        st.error("\u5bc6\u7801\u9519\u8bef")
    st.stop()

PSD_EXTENSIONS = (".psd", ".psb")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
VECTOR_EXTENSIONS = (".pdf", ".eps", ".ai")
ALL_TEMPLATE_TYPES = [e.lstrip(".") for e in PSD_EXTENSIONS + IMAGE_EXTENSIONS + VECTOR_EXTENSIONS]

LIST_EXTENSIONS = ["csv", "xlsx", "xls"]

# ── helpers ──────────────────────────────────────────────


@st.cache_data
def scan_fonts():
    """Scan bundled fonts dir + common system font dirs."""
    fonts = {}
    search_dirs = [str(FONTS_DIR)]
    for sys_dir in ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
                    "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
                    "/usr/share/fonts", "/usr/local/share/fonts"]:
        if os.path.isdir(sys_dir):
            search_dirs.append(sys_dir)

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not any(f.lower().endswith(ext) for ext in (".ttf", ".otf", ".ttc")):
                continue
            path = os.path.join(d, f)
            try:
                family, style = ImageFont.truetype(path, 20).getname()
                display = f"{family} ({style})"
                fonts[display] = path
            except Exception:
                pass
    return dict(sorted(fonts.items()))


def get_default_font_path():
    bundled = FONTS_DIR / "OPPOSans4.ttf"
    if bundled.exists():
        return str(bundled)
    fonts = scan_fonts()
    if fonts:
        return next(iter(fonts.values()))
    return None


def file_suffix(uploaded):
    return Path(uploaded.name).suffix.lower()


def load_psd(uploaded):
    suffix = file_suffix(uploaded)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        return PSDImage.open(tmp.name)


def load_image(uploaded):
    suffix = file_suffix(uploaded)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        try:
            img = Image.open(tmp.name).convert("RGBA")
            return img
        except Exception as e:
            st.error(f"无法打开此文件格式。错误: {e}")
            return None


def get_text_layers(psd):
    return [l for l in psd.descendants() if l.kind == "type"]


def get_qr_layer(psd):
    for l in psd.descendants():
        if l.kind == "smartobject" and "二维码" in l.name:
            return l
    return None


def detect_qr_region(psd):
    qr_layer = get_qr_layer(psd)
    if qr_layer is not None:
        return (qr_layer.left, qr_layer.top, qr_layer.right, qr_layer.bottom)
    return None


def extract_qr_mask(psd):
    """Extract the real alpha mask from the QR smart object layer."""
    qr_layer = get_qr_layer(psd)
    if qr_layer is None:
        return None
    try:
        layer_img = qr_layer.composite()
        if layer_img and layer_img.mode == "RGBA":
            return layer_img.split()[3]
    except Exception:
        pass
    return None


def composite_background(psd):
    for l in psd.descendants():
        if l.kind == "type":
            l.visible = False
    return psd.composite()


def rounded_corner_mask(size, radius):
    mask = Image.new("L", size, 255)
    d = ImageDraw.Draw(mask)
    r = radius
    d.rectangle([0, 0, r, r], fill=0)
    d.rectangle([size[0] - r, 0, size[0], r], fill=0)
    d.rectangle([0, size[1] - r, r, size[1]], fill=0)
    d.rectangle([size[0] - r, size[1] - r, size[0], size[1]], fill=0)
    d.pieslice([0, 0, r * 2, r * 2], 180, 270, fill=255)
    d.pieslice([size[0] - r * 2, 0, size[0], r * 2], 270, 360, fill=255)
    d.pieslice([0, size[1] - r * 2, r * 2, size[1]], 90, 180, fill=255)
    d.pieslice([size[0] - r * 2, size[1] - r * 2, size[0], size[1]], 0, 90, fill=255)
    return mask


def replace_qr(background, qr_image, qr_box, real_mask=None, corner_radius=3):
    tw = qr_box[2] - qr_box[0]
    th = qr_box[3] - qr_box[1]
    qr_resized = qr_image.convert("RGBA").resize((tw, th), Image.LANCZOS)
    if real_mask is not None:
        mask = real_mask.resize((tw, th), Image.LANCZOS)
    else:
        mask = rounded_corner_mask((tw, th), corner_radius)
    bg_region = background.crop(qr_box).convert("RGBA")
    composite = Image.composite(qr_resized, bg_region, mask)
    background.paste(composite, (qr_box[0], qr_box[1]))
    return background


def calibrate_font_size(font_path, text, target_height, raw_size):
    """Find Pillow font size where text height matches PSD layer height."""
    best_size = int(raw_size)
    best_diff = 999
    for s in range(int(raw_size), int(raw_size) + 20):
        f = ImageFont.truetype(font_path, s)
        bb = f.getbbox(text)
        h = bb[3] - bb[1]
        diff = abs(h - target_height)
        if diff < best_diff:
            best_diff = diff
            best_size = s
        if h >= target_height:
            return s
    return best_size


def get_font_color(psd):
    """Extract text color from PSD."""
    for l in psd.descendants():
        if l.kind == "type":
            ss = l.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
            color_vals = ss.get("FillColor", {}).get("Values", [1.0, 1.0, 1.0, 1.0])
            return tuple(int(v * 255) for v in color_vals)
    return (255, 255, 255, 255)


def get_text_layer_positions(psd, font_path=None):
    """Return {layer_name: (center_y, calibrated_font_size, psd_width, stroke_width)}.

    stroke_width is derived from PSD FauxBold flag or bold font variant name,
    simulating Photoshop's bold rendering in Pillow via stroke.
    """
    positions = {}
    for l in psd.descendants():
        if l.kind == "type":
            cy = (l.top + l.bottom) // 2
            ss = l.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
            raw_size = ss.get("FontSize", 51)
            cal_size = int(raw_size)
            if font_path and l.height > 0 and len(l.text) >= 1:
                cal_size = calibrate_font_size(font_path, l.text, l.height, raw_size)

            stroke_w = 0
            faux_bold = ss.get("FauxBold", False)
            if faux_bold:
                stroke_w = max(1, round(cal_size / 36))

            if stroke_w == 0:
                try:
                    fs = l.engine_dict.get("ResourceDict", {}).get("FontSet", [])
                    font_idx = int(ss.get("Font", 0))
                    if font_idx < len(fs):
                        ps_name = fs[font_idx].get("Name", "").lower()
                        if any(w in ps_name for w in
                               ["bold", "heavy", "black", "semibold", "demibold"]):
                            stroke_w = max(1, round(cal_size / 36))
                except Exception:
                    pass

            positions[l.name] = (cy, cal_size, l.width, stroke_w)
    return positions


import re as _re
_EMOJI_RE = _re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F\U0000200D]+")


def _has_emoji(text):
    return bool(_EMOJI_RE.search(text))


def draw_centered_text(draw, font, text, center_y, img_width, color, stroke_width=0):
    """Draw text horizontally and vertically centered using standard draw.text."""
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    x = (img_width - text_w) // 2
    y = center_y - (bbox[1] + bbox[3]) // 2
    if stroke_width > 0:
        draw.text((x, y), text, font=font, fill=color,
                  stroke_width=stroke_width, stroke_fill=color)
    else:
        draw.text((x, y), text, font=font, fill=color)


def _draw_centered_emoji(img, font, text, center_y, img_width, color, stroke_width=0):
    """Draw text with emoji support using pilmoji."""
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    x = (img_width - text_w) // 2
    y = center_y - (bbox[1] + bbox[3]) // 2
    with Pilmoji(img) as pmj:
        if stroke_width > 0:
            pmj.text((x, y), text, font=font, fill=color,
                     stroke_width=stroke_width, stroke_fill=color)
        else:
            pmj.text((x, y), text, font=font, fill=color)


def generate_one(background, text_items, img_width, color, font_path):
    """text_items: list of (text_str, center_y, font_size[, stroke_width])."""
    img = background.copy()
    draw = ImageDraw.Draw(img)
    use_emoji = any(_has_emoji(item[0]) for item in text_items if item[0])
    for item in text_items:
        text, cy, fsize = item[0], item[1], item[2]
        sw = item[3] if len(item) > 3 else 0
        if text:
            f = ImageFont.truetype(font_path, fsize)
            if use_emoji and _has_emoji(text):
                _draw_centered_emoji(img, f, text, cy, img_width, color, stroke_width=sw)
            else:
                draw_centered_text(draw, f, text, cy, img_width, color, stroke_width=sw)
    return img


def check_image_quality(img, text_items, img_width, qr_box, font_path):
    """Quality checks: text bounds, spacing vs PSD width, QR readability."""
    issues = []

    for item in text_items:
        text, center_y, fsize = item[0], item[1], item[2]
        if not text:
            continue
        f = ImageFont.truetype(font_path, fsize)
        bbox = f.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (img_width - text_w) // 2
        draw_y = center_y - (bbox[1] + bbox[3]) // 2
        if x < 0:
            issues.append(("error", f"\u6587\u5b57\u300c{text}\u300d\u8d85\u51fa\u56fe\u7247\u5bbd\u5ea6"))
        elif x < 20:
            issues.append(("warning", f"\u6587\u5b57\u300c{text}\u300d\u8ddd\u8fb9\u7f18\u592a\u8fd1 ({int(x)}px)"))
        if draw_y < 0 or draw_y + text_h > img.size[1]:
            issues.append(("error", f"\u6587\u5b57\u300c{text}\u300d\u8d85\u51fa\u56fe\u7247\u9ad8\u5ea6"))

    if qr_box:
        try:
            pad = 10
            crop_box = (max(0, qr_box[0] - pad), max(0, qr_box[1] - pad),
                        min(img.width, qr_box[2] + pad), min(img.height, qr_box[3] + pad))
            qr_crop = img.crop(crop_box).convert("RGB")
            qr_crop = qr_crop.resize((qr_crop.width * 2, qr_crop.height * 2), Image.LANCZOS)
            arr = np.array(qr_crop)
            detector = cv2.QRCodeDetector()
            data, det_bbox, _ = detector.detectAndDecode(arr)
            if data:
                issues.append(("success", "\u4e8c\u7ef4\u7801\u53ef\u8bc6\u522b"))
            else:
                issues.append(("warning", "\u4e8c\u7ef4\u7801\u53ef\u80fd\u65e0\u6cd5\u8bc6\u522b, \u5efa\u8bae\u624b\u673a\u626b\u7801\u786e\u8ba4"))
        except Exception:
            pass

    return issues


def compare_preview_quality(original_img, preview_img, text_items, img_width, qr_box, font_path, use_custom_font=False):
    """Compare original vs preview for diff-based quality checks."""
    issues = []

    # 2) 字体应用是否符合指定字体策略
    font_name = Path(font_path).name.lower() if font_path else ""
    if use_custom_font:
        issues.append(("success", f"已应用自定义字体: {Path(font_path).name}"))
    else:
        if "oppo" in font_name:
            issues.append(("success", "字体检查通过：已使用 OPPO 字体"))
        else:
            issues.append(("warning", f"字体可能不是指定 OPPO 字体（当前: {Path(font_path).name}）"))

    # 1) 字体大小、字距、边距、居中、对齐 + 5) 空格
    for item in text_items:
        text, center_y, fsize = item[0], item[1], item[2]
        if not text:
            issues.append(("warning", "检测到空文本，可能导致内容缺失"))
            continue

        if text != text.strip():
            issues.append(("warning", f"文本「{text}」存在首尾空格"))
        if "  " in text:
            issues.append(("warning", f"文本「{text}」存在连续空格"))

        f = ImageFont.truetype(font_path, fsize)
        bbox = f.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (img_width - text_w) // 2
        y = center_y - (bbox[1] + bbox[3]) // 2
        left_gap = x
        right_gap = img_width - (x + text_w)
        if abs(left_gap - right_gap) > 6:
            issues.append(("warning", f"文本「{text}」左右边距不平衡（差值 {abs(left_gap-right_gap)}px）"))
        draw_center = y + text_h / 2
        if abs(draw_center - center_y) > 3:
            issues.append(("warning", f"文本「{text}」垂直对齐偏差 {abs(draw_center-center_y):.1f}px"))

    # 3) 背景清晰和细节是否变化
    try:
        ori_rgb = np.array(original_img.convert("RGB"), dtype=np.int16)
        pre_rgb = np.array(preview_img.convert("RGB"), dtype=np.int16)
        if ori_rgb.shape == pre_rgb.shape:
            h, w, _ = ori_rgb.shape
            mask = np.ones((h, w), dtype=np.uint8)

            for item in text_items:
                text, center_y, fsize = item[0], item[1], item[2]
                if not text:
                    continue
                f = ImageFont.truetype(font_path, fsize)
                bbox = f.getbbox(text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x = max(0, (img_width - text_w) // 2 - 18)
                y = max(0, int(center_y - (bbox[1] + bbox[3]) // 2) - 18)
                x2 = min(w, x + text_w + 36)
                y2 = min(h, y + text_h + 36)
                mask[y:y2, x:x2] = 0

            if qr_box:
                x1, y1, x2, y2 = qr_box
                x1 = max(0, x1 - 16)
                y1 = max(0, y1 - 16)
                x2 = min(w, x2 + 16)
                y2 = min(h, y2 + 16)
                mask[y1:y2, x1:x2] = 0

            diff = np.abs(pre_rgb - ori_rgb).mean(axis=2)
            bg_diff = float(diff[mask == 1].mean()) if np.any(mask == 1) else 0.0
            if bg_diff > 7.5:
                issues.append(("warning", f"背景差异偏大（均值 {bg_diff:.2f}），可能影响清晰度或细节"))
            else:
                issues.append(("success", "背景差异检查通过"))

            ori_gray = cv2.cvtColor(ori_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            pre_gray = cv2.cvtColor(pre_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            ori_sharp = cv2.Laplacian(ori_gray, cv2.CV_64F).var()
            pre_sharp = cv2.Laplacian(pre_gray, cv2.CV_64F).var()
            if ori_sharp > 0 and pre_sharp / ori_sharp < 0.82:
                issues.append(("warning", f"替换图清晰度下降（{pre_sharp/ori_sharp:.2f}x）"))
    except Exception:
        issues.append(("warning", "背景清晰度检查失败，请人工放大复核细节"))

    # 4) 二维码三重校验：图形结构 + 清晰度 + 乱码
    if qr_box:
        try:
            qx1, qy1, qx2, qy2 = qr_box

            ori_qr = original_img.crop((qx1, qy1, qx2, qy2)).convert("L")
            pre_qr = preview_img.crop((qx1, qy1, qx2, qy2)).convert("L")
            ori_arr = np.array(ori_qr, dtype=np.float64)
            pre_arr = np.array(pre_qr.resize(ori_qr.size, Image.LANCZOS), dtype=np.float64)

            # 4a) 图形结构校对：与源文件 1:1 像素对比（SSIM 简化版）
            mu_o, mu_p = ori_arr.mean(), pre_arr.mean()
            sig_o = ori_arr.std()
            sig_p = pre_arr.std()
            cov = ((ori_arr - mu_o) * (pre_arr - mu_p)).mean()
            c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ssim = ((2 * mu_o * mu_p + c1) * (2 * cov + c2)) / \
                   ((mu_o ** 2 + mu_p ** 2 + c1) * (sig_o ** 2 + sig_p ** 2 + c2))
            if ssim > 0.85:
                issues.append(("success", f"\u4e8c\u7ef4\u7801\u56fe\u5f62\u7ed3\u6784\u4e0e\u6e90\u6587\u4ef6\u4e00\u81f4 (SSIM={ssim:.3f})"))
            elif ssim > 0.6:
                issues.append(("warning", f"\u4e8c\u7ef4\u7801\u56fe\u5f62\u4e0e\u6e90\u6587\u4ef6\u5b58\u5728\u5dee\u5f02 (SSIM={ssim:.3f})\uff0c\u5efa\u8bae\u68c0\u67e5\u5706\u89d2/\u8fb9\u7f18"))
            else:
                issues.append(("error", f"\u4e8c\u7ef4\u7801\u56fe\u5f62\u4e0e\u6e90\u6587\u4ef6\u5dee\u5f02\u8fc7\u5927 (SSIM={ssim:.3f})\uff0c\u53ef\u80fd\u88ab\u7834\u574f"))

            # 4b) 清晰度校对
            pre_qr_rgb = np.array(preview_img.crop((qx1, qy1, qx2, qy2)).convert("RGB"))
            qr_gray = cv2.cvtColor(pre_qr_rgb, cv2.COLOR_RGB2GRAY)
            qr_sharp = cv2.Laplacian(qr_gray, cv2.CV_64F).var()
            ori_qr_rgb = np.array(original_img.crop((qx1, qy1, qx2, qy2)).convert("RGB"))
            ori_qr_gray = cv2.cvtColor(ori_qr_rgb, cv2.COLOR_RGB2GRAY)
            ori_qr_sharp = cv2.Laplacian(ori_qr_gray, cv2.CV_64F).var()
            if ori_qr_sharp > 0:
                ratio = qr_sharp / ori_qr_sharp
                if ratio > 0.85:
                    issues.append(("success", f"\u4e8c\u7ef4\u7801\u6e05\u6670\u5ea6\u6b63\u5e38 ({ratio:.2f}x)"))
                elif ratio > 0.5:
                    issues.append(("warning", f"\u4e8c\u7ef4\u7801\u6e05\u6670\u5ea6\u4e0b\u964d ({ratio:.2f}x)\uff0c\u53ef\u80fd\u5f71\u54cd\u626b\u7801"))
                else:
                    issues.append(("error", f"\u4e8c\u7ef4\u7801\u6e05\u6670\u5ea6\u4e25\u91cd\u4e0b\u964d ({ratio:.2f}x)"))
            elif qr_sharp < 45:
                issues.append(("warning", "\u4e8c\u7ef4\u7801\u6e05\u6670\u5ea6\u504f\u4f4e"))

            # 4c) 乱码校对：QR 解码 + 定位方块完整性
            qr_big = cv2.resize(pre_qr_rgb, (pre_qr_rgb.shape[1] * 2, pre_qr_rgb.shape[0] * 2),
                                interpolation=cv2.INTER_LANCZOS4)
            detector = cv2.QRCodeDetector()
            data, det_pts, _ = detector.detectAndDecode(qr_big)
            if data:
                issues.append(("success", "\u4e8c\u7ef4\u7801\u53ef\u6b63\u5e38\u89e3\u7801\uff0c\u65e0\u4e71\u7801"))
            else:
                if det_pts is not None:
                    issues.append(("warning", "\u4e8c\u7ef4\u7801\u5b9a\u4f4d\u6846\u53ef\u68c0\u6d4b\u4f46\u89e3\u7801\u5931\u8d25\uff0c\u5efa\u8bae\u624b\u673a\u626b\u7801\u786e\u8ba4"))
                else:
                    issues.append(("error", "\u4e8c\u7ef4\u7801\u5b9a\u4f4d\u6846\u672a\u68c0\u6d4b\u5230\uff0c\u53ef\u80fd\u5b58\u5728\u4e25\u91cd\u4e71\u7801\u6216\u88c1\u5207"))
        except Exception:
            issues.append(("warning", "\u4e8c\u7ef4\u7801\u8be6\u7ec6\u68c0\u67e5\u5931\u8d25\uff0c\u8bf7\u4eba\u5de5\u626b\u7801\u590d\u6838"))

    # 6) 图片是否乱码（全图异常波动）
    try:
        ori = np.array(original_img.convert("RGB"), dtype=np.int16)
        pre = np.array(preview_img.convert("RGB"), dtype=np.int16)
        if ori.shape == pre.shape:
            global_diff = float(np.abs(pre - ori).mean())
            if global_diff > 20:
                issues.append(("warning", f"全图差异偏大（{global_diff:.2f}），可能存在异常渲染或乱码"))
    except Exception:
        pass

    return issues


def parse_spreadsheet(uploaded):
    suffix = file_suffix(uploaded)
    try:
        if suffix == ".csv":
            text = uploaded.getvalue().decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(text))
        elif suffix == ".xlsx":
            df = pd.read_excel(uploaded, engine="openpyxl")
        elif suffix == ".xls":
            df = pd.read_excel(uploaded, engine="xlrd")
        elif suffix == ".et":
            st.error("WPS .et 格式暂不支持，请另存为 .xlsx 后重新上传。")
            return [], []
        else:
            st.error(f"不支持的名单格式: {suffix}")
            return [], []
    except Exception as e:
        st.error(f"读取名单失败: {e}")
        return [], []

    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    fields = list(df.columns)
    rows = df.astype(str).to_dict("records")
    return rows, fields


# ── UI ───────────────────────────────────────────────────

st.markdown('<div class="apple-section-title">第一步：上传文件</div>', unsafe_allow_html=True)

upload_col1, upload_col2, upload_col3 = st.columns(3)
with upload_col1:
    template_file = st.file_uploader(
        "1. 上传模板文件",
        type=ALL_TEMPLATE_TYPES,
        help="PSD / PSB / PNG / JPG / TIFF / BMP / WebP / PDF / EPS / AI",
    )
    st.markdown(
        '<div class="apple-info-card"><strong>模板文件规则</strong>'
        '<span>支持 PSD、PSB、PNG、JPG、AI、PDF 等格式。建议优先使用原始设计稿，图层识别更准确。</span>'
        '<br><span style="font-size:0.84rem;color:rgba(128,128,132,0.75);">如需修改模板，重新上传即可覆盖当前文件。</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with upload_col2:
    list_file = st.file_uploader(
        "2. 上传名单",
        type=LIST_EXTENSIONS,
        help="CSV / Excel (.xlsx) / Excel (.xls)",
    )
    st.markdown(
        '<div class="apple-info-card"><strong>名单规则</strong>'
        '<span>支持 CSV、XLSX、XLS。建议至少包含“公司名”和“人名”字段，自动识别更准确。</span>'
        '<br><span style="font-size:0.84rem;color:rgba(128,128,132,0.75);">如需修正数据，重新上传名单文件即可生效。</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with upload_col3:
    qr_file = st.file_uploader(
        "3. 上传替换二维码（可选）",
        type=["png", "jpg", "jpeg", "webp"],
    )
    st.markdown(
        '<div class="apple-info-card"><strong>二维码规则</strong>'
        '<span>可选项。不上传则保留模板原二维码；上传后按模板位置自动替换并保留圆角效果。</span>'
        '<br><span style="font-size:0.84rem;color:rgba(128,128,132,0.75);">如需更换二维码，重新上传图片即可覆盖。</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.caption("上传方式：可拖拽文件到上传框，或点击“选择文件”按钮上传。")

if template_file and list_file:
    suffix = file_suffix(template_file)
    is_psd = suffix in PSD_EXTENSIONS

    with st.spinner("正在解析模板..."):
        if is_psd:
            psd = load_psd(template_file)
            text_layers = get_text_layers(psd)
            layer_names = [l.name for l in text_layers]
            _default_font = str(FONTS_DIR / "OPPOSans4.ttf")
            positions = get_text_layer_positions(psd, _default_font)
            font_color = get_font_color(psd)
            font_size = 51
            qr_box = detect_qr_region(psd)
            qr_real_mask = extract_qr_mask(psd)
            original_img = psd.composite()
            bg = composite_background(psd)
            img_width = psd.width
            img_height = psd.height
        else:
            loaded = load_image(template_file)
            if loaded is None:
                st.stop()
            original_img = loaded.copy()
            bg = loaded
            img_width, img_height = bg.size
            font_size = 51
            font_color = (255, 255, 255, 255)
            qr_box = None
            qr_real_mask = None
            layer_names = []
            positions = {}

    rows, fields = parse_spreadsheet(list_file)
    if not rows:
        st.warning("名单为空或读取失败，请检查文件。")
        st.stop()

    info_parts = []
    if is_psd:
        info_parts.append(f"检测到 {len(text_layers)} 个文字图层")
    else:
        info_parts.append(f"模板尺寸 {img_width}x{img_height}")
    info_parts.append(f"名单共 {len(rows)} 条记录")
    st.success(f"解析完成：{'，'.join(info_parts)}")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("名单记录数", len(rows))
    with m2:
        st.metric("模板宽度", img_width)

    # ── auto-detect fields ──
    COMPANY_KEYWORDS = ["company", "\u516c\u53f8", "\u5355\u4f4d", "\u673a\u6784", "\u4f01\u4e1a", "\u7ec4\u7ec7"]
    NAME_KEYWORDS = ["name", "\u59d3\u540d", "\u4eba\u540d", "\u79f0\u547c", "\u540d\u5b57"]

    def auto_detect_field(fields, keywords):
        for i, f in enumerate(fields):
            for kw in keywords:
                if kw in f.lower():
                    return i
        return None

    def auto_detect_layer(layer_names, keywords):
        for i, name in enumerate(layer_names):
            for kw in keywords:
                if kw in name.lower():
                    return i
        return None

    company_idx = auto_detect_field(fields, COMPANY_KEYWORDS)
    name_idx = auto_detect_field(fields, NAME_KEYWORDS)
    has_company_field = company_idx is not None
    has_name_field = name_idx is not None

    psd_company_layer_idx = auto_detect_layer(layer_names, COMPANY_KEYWORDS) if is_psd else None
    psd_name_layer_idx = auto_detect_layer(layer_names, NAME_KEYWORDS) if is_psd else None

    # ── field mapping UI ──
    st.markdown('<div class="apple-section-title">第二步：字段映射与样式设置</div>', unsafe_allow_html=True)

    if has_company_field and has_name_field:
        st.info(f"\u5df2\u81ea\u52a8\u8bc6\u522b: \u516c\u53f8\u540d=\u300c{fields[company_idx]}\u300d, \u4eba\u540d=\u300c{fields[name_idx]}\u300d")
    elif has_name_field and not has_company_field:
        st.info(f"\u5df2\u81ea\u52a8\u8bc6\u522b: \u4eba\u540d=\u300c{fields[name_idx]}\u300d (\u672a\u68c0\u6d4b\u5230\u516c\u53f8\u540d\u5b57\u6bb5)")
    elif has_company_field and not has_name_field:
        st.info(f"\u5df2\u81ea\u52a8\u8bc6\u522b: \u516c\u53f8\u540d=\u300c{fields[company_idx]}\u300d (\u672a\u68c0\u6d4b\u5230\u4eba\u540d\u5b57\u6bb5)")

    st.caption("\u6839\u636e\u4f60\u7684\u540d\u5355\u5185\u5bb9\u52fe\u9009\u9700\u8981\u66ff\u6362\u7684\u5b57\u6bb5\u3002\u672a\u52fe\u9009\u7684\u5b57\u6bb5\u4e0d\u4f1a\u66ff\u6362\uff0c\u4fdd\u6301\u539f\u6a21\u677f\u5185\u5bb9\u3002")
    enable_company = st.checkbox("\u542f\u7528\u516c\u53f8\u540d", value=has_company_field,
                                 help="\u52fe\u9009\u540e\u4f1a\u7528\u540d\u5355\u4e2d\u7684\u516c\u53f8\u540d\u66ff\u6362\u6a21\u677f\u4e2d\u7684\u516c\u53f8\u540d\u4f4d\u7f6e")
    enable_name = st.checkbox("\u542f\u7528\u4eba\u540d", value=has_name_field,
                              help="\u52fe\u9009\u540e\u4f1a\u7528\u540d\u5355\u4e2d\u7684\u4eba\u540d\u66ff\u6362\u6a21\u677f\u4e2d\u7684\u4eba\u540d\u4f4d\u7f6e")

    if not enable_company and not enable_name:
        st.warning("\u81f3\u5c11\u9700\u8981\u542f\u7528\u4e00\u4e2a\u5b57\u6bb5")
        st.stop()

    company_field = None
    company_y = 0
    company_fsize = font_size
    company_stroke = 0
    name_field = None
    name_y = 0
    name_fsize = font_size
    name_stroke = 0
    company_layer = None
    name_layer = None

    if enable_company:
        mcol1, mcol2 = st.columns(2) if enable_name else [st.container(), None]
        with mcol1:
            company_field = st.selectbox(
                "\u516c\u53f8\u540d\u5bf9\u5e94\u5b57\u6bb5", fields,
                index=company_idx if company_idx is not None else 0)
            if is_psd and layer_names:
                cl_idx = psd_company_layer_idx if psd_company_layer_idx is not None else max(0, len(layer_names) - 1)
                company_layer = st.selectbox("\u516c\u53f8\u540d\u5bf9\u5e94 PSD \u56fe\u5c42", layer_names, index=cl_idx)
                company_y = positions[company_layer][0]
                company_fsize = positions[company_layer][1]
                company_stroke = positions[company_layer][3]
            else:
                company_y = st.number_input("\u516c\u53f8\u540d Y \u5750\u6807", 0, img_height, int(img_height * 0.45))

    if enable_name:
        col_ctx = mcol2 if (enable_company and mcol2 is not None) else st.container()
        with col_ctx:
            name_field = st.selectbox(
                "\u4eba\u540d\u5bf9\u5e94\u5b57\u6bb5", fields,
                index=name_idx if name_idx is not None else min(1, len(fields) - 1))
            if is_psd and layer_names:
                nl_idx = psd_name_layer_idx if psd_name_layer_idx is not None else 0
                name_layer = st.selectbox("\u4eba\u540d\u5bf9\u5e94 PSD \u56fe\u5c42", layer_names, index=nl_idx)
                name_y = positions[name_layer][0]
                name_fsize = positions[name_layer][1]
                name_stroke = positions[name_layer][3]
            else:
                name_y = st.number_input("\u4eba\u540d Y \u5750\u6807", 0, img_height, int(img_height * 0.48))

    # ── error detection ──
    mapping_ok = True
    if enable_company and enable_name:
        if company_field == name_field:
            st.error("\u516c\u53f8\u540d\u548c\u4eba\u540d\u9009\u4e86\u540c\u4e00\u4e2a\u5b57\u6bb5\uff0c\u8bf7\u4fee\u6539!")
            mapping_ok = False
        if is_psd and company_layer and name_layer and company_layer == name_layer:
            st.error("\u516c\u53f8\u540d\u548c\u4eba\u540d\u9009\u4e86\u540c\u4e00\u4e2a PSD \u56fe\u5c42\uff0c\u8bf7\u4fee\u6539!")
            mapping_ok = False

    if mapping_ok and enable_name and name_field:
        sample_names = [rows[i][name_field] for i in range(min(3, len(rows)))]
        long_names = [n for n in sample_names if len(n) > 4]
        if long_names:
            st.warning(f"\u4eba\u540d\u5b57\u6bb5\u4e2d\u53d1\u73b0\u8f83\u957f\u7684\u503c: \u300c{'、'.join(long_names)}\u300d\uff0c\u8bf7\u786e\u8ba4\u662f\u5426\u9009\u5bf9\u4e86\u5b57\u6bb5")

    # ── mapping preview table ──
    if mapping_ok:
        st.markdown("**\u6620\u5c04\u9884\u89c8 (\u524d3\u884c):**")
        preview_data = []
        for i in range(min(3, len(rows))):
            row_preview = {}
            if enable_company and company_field:
                row_preview["\u516c\u53f8\u540d"] = rows[i][company_field]
            if enable_name and name_field:
                row_preview["\u4eba\u540d"] = rows[i][name_field]
            preview_data.append(row_preview)
        if preview_data:
            st.table(preview_data)

    if not mapping_ok:
        st.stop()

    # ── font selection ──
    st.markdown("### 字体选择")

    custom_font_file = st.file_uploader(
        "上传自定义字体 (可选, 支持 .ttf / .otf)",
        type=["ttf", "otf"],
        help="不上传则使用默认字体 OPPO Sans 4.0",
    )

    if custom_font_file:
        font_tmp = tempfile.NamedTemporaryFile(suffix=file_suffix(custom_font_file), delete=False)
        font_tmp.write(custom_font_file.getvalue())
        font_tmp.flush()
        font_path = font_tmp.name
        try:
            family, style = ImageFont.truetype(font_path, 20).getname()
            st.success(f"已加载自定义字体：{family} ({style})")
        except Exception as e:
            st.error(f"字体文件无法加载: {e}")
            font_path = get_default_font_path()
    else:
        all_fonts = scan_fonts()
        font_names = list(all_fonts.keys())
        if font_names:
            default_idx = 0
            for i, name in enumerate(font_names):
                if "OPPO Sans 4.0" in name:
                    default_idx = i
                    break
            selected_font = st.selectbox(
                "或从已有字体中选择",
                font_names,
                index=default_idx,
                help=f"已扫描到 {len(font_names)} 个可用字体",
            )
            font_path = all_fonts[selected_font]
        else:
            font_path = get_default_font_path()
            if font_path:
                st.info("使用默认字体 OPPO Sans 4.0")
            else:
                st.error("未找到任何可用字体，请上传 .ttf 字体文件")
                st.stop()

    if not is_psd:
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            font_size = st.number_input("字号", 10, 200, int(font_size))
        with fcol2:
            color_hex = st.color_picker("文字颜色", "#FFFFFF")
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            font_color = (r, g, b, 255)

    # ── QR replacement ──
    if qr_file and qr_box:
        qr_img = Image.open(qr_file).convert("RGBA")
        bg = replace_qr(bg, qr_img, qr_box, real_mask=qr_real_mask)

    def build_text_items(row):
        items = []
        if enable_company and company_field:
            items.append((row[company_field], company_y, company_fsize, company_stroke))
        if enable_name and name_field:
            items.append((row[name_field], name_y, name_fsize, name_stroke))
        return items

    def build_filename(row):
        parts = []
        if enable_company and company_field:
            parts.append(row[company_field])
        if enable_name and name_field:
            parts.append(row[name_field])
        return "_".join(parts) if parts else f"row"

    # ── preview: original vs first ──
    st.markdown('<div class="apple-section-title">第三步：预览与生成</div>', unsafe_allow_html=True)
    first = rows[0]
    preview = generate_one(bg, build_text_items(first), img_width, font_color, font_path)
    preview_show = st.session_state.get("regen_preview", preview)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.image(original_img, caption="\u539f\u59cb\u6a21\u677f", use_container_width=True)
    with pcol2:
        st.image(preview_show, caption=f"\u66ff\u6362\u6548\u679c: {build_filename(first)}",
                 use_container_width=True)

    st.markdown("---")
    st.caption("\u8bf7\u4ed4\u7ec6\u5bf9\u6bd4\u4e0a\u65b9\u4e24\u5f20\u56fe\uff0c\u786e\u8ba4\u5b57\u4f53\u5927\u5c0f\u3001\u4f4d\u7f6e\u3001\u95f4\u8ddd\u3001\u4e8c\u7ef4\u7801\u662f\u5426\u4e0e\u539f\u56fe\u4e00\u81f4\u3002")

    if "preview_confirmed" not in st.session_state:
        st.session_state.preview_confirmed = False
    if "single_check_done" not in st.session_state:
        st.session_state.single_check_done = False
    if "single_check_issues" not in st.session_state:
        st.session_state.single_check_issues = []
    if "checked_report" not in st.session_state:
        st.session_state.checked_report = ""

    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        st.markdown(
            '<div class="action-card"><strong>路径 A：确认无问题</strong><span>未填写问题时，可直接进入下一步。</span></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "✅ 无问题，直接下一步",
            type="primary",
            use_container_width=True,
            key="btn_preview_direct_next",
        ):
            report_text = st.session_state.get("preview_report", "").strip()
            if report_text:
                st.warning("你已填写问题描述，请先点击“有错误点击重新生成预览”，确认无问题后再进入下一步。")
                st.session_state.preview_confirmed = False
            else:
                st.session_state.preview_confirmed = True
                st.rerun()
    with confirm_col2:
        st.markdown(
            '<div class="action-card"><strong>路径 B：发现问题先修复</strong><span>输入问题后，先重新生成预览并检查，再决定是否进入下一步。</span></div>',
            unsafe_allow_html=True,
        )
        report = st.text_input(
            "请描述发现的问题",
            placeholder="\u4f8b\u5982: \u5b57\u4f53\u504f\u5c0f / \u4f4d\u7f6e\u504f\u79fb / \u95f4\u8ddd\u4e0d\u5bf9...",
            key="preview_report",
        )
        report_text = report.strip()

        if report_text != st.session_state.get("checked_report", ""):
            st.session_state.single_check_done = False
            st.session_state.single_check_issues = []

        if st.button(
            "有错误点击重新生成预览",
            use_container_width=True,
            key="btn_preview_regen_check",
        ):
            if report_text:
                regen_img = generate_one(bg, build_text_items(first), img_width, font_color, font_path)
                st.session_state["regen_preview"] = regen_img
                text_items_now = build_text_items(first)
                basic_issues = check_image_quality(regen_img, text_items_now, img_width, qr_box, font_path)
                diff_issues = compare_preview_quality(
                    original_img,
                    regen_img,
                    text_items_now,
                    img_width,
                    qr_box,
                    font_path,
                    use_custom_font=bool(custom_font_file),
                )
                st.session_state.single_check_issues = basic_issues + diff_issues
                st.session_state.single_check_done = True
                st.session_state.checked_report = report_text
                st.session_state.preview_confirmed = False
                st.rerun()
            else:
                st.warning("\u8bf7\u5148\u5728\u4e0a\u65b9\u8f93\u5165\u95ee\u9898\u63cf\u8ff0\uff0c\u518d\u70b9\u51fb\u91cd\u65b0\u751f\u6210\u3002")
        st.caption(
            "操作提示：先描述问题，再点“有错误点击重新生成预览”。检查通过后再进入下一步。"
        )
        if report_text:
            st.warning(f"\u4f60\u53cd\u9988\u7684\u95ee\u9898: \u300c{report}\u300d")
            if st.session_state.single_check_done and st.session_state.get("checked_report", "") == report_text:
                has_errors = False
                has_warnings = False
                if not st.session_state.single_check_issues:
                    st.success("本次检查未发现问题。")
                for level, msg in st.session_state.single_check_issues:
                    if level == "error":
                        has_errors = True
                        st.error(msg)
                    elif level == "warning":
                        has_warnings = True
                        st.warning(msg)
                    elif level == "success":
                        st.success(msg)

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button(
                        "有问题继续生成修复预览",
                        use_container_width=True,
                        key="btn_preview_continue_fix",
                    ):
                        st.session_state.single_check_done = False
                        st.session_state.preview_confirmed = False
                        st.rerun()
                with action_col2:
                    if st.button(
                        "无问题，进行下一步",
                        type="primary",
                        use_container_width=True,
                        key="btn_preview_no_issue_next",
                    ):
                        if has_errors:
                            st.error("仍存在错误项，请继续修复后再进行下一步。")
                            st.session_state.preview_confirmed = False
                        else:
                            st.session_state.preview_confirmed = True
                            st.rerun()
            else:
                st.info("请先点击“有错误点击重新生成预览”，确认问题是否已解决。")

    if not st.session_state.preview_confirmed:
        if st.session_state.get("preview_report", "").strip():
            st.info("已记录问题，请先点击“有错误点击重新生成预览”；确认无问题后点击“无问题，进行下一步”。")
        else:
            st.info("\u8bf7\u5148\u786e\u8ba4\u4e0a\u65b9\u9884\u89c8\u6548\u679c\u65e0\u8bef\uff0c\u624d\u80fd\u7ee7\u7eed\u4e0b\u4e00\u6b65")
        st.stop()

    # ── step 1: preview samples ──
    st.markdown("### \u751f\u6210\u9884\u89c8")
    total = len(rows)
    preview_count = st.radio(
        "\u9009\u62e9\u9884\u89c8\u6570\u91cf",
        [5, 10],
        horizontal=True,
        format_func=lambda x: f"\u9884\u89c8\u524d {x} \u5f20",
    )
    preview_count = min(preview_count, total)

    if st.button("\u751f\u6210\u9884\u89c8", type="secondary", use_container_width=True):
        preview_imgs = []
        all_issues = []
        st.session_state["preview_gallery_confirmed"] = False
        progress = st.progress(0, text="\u6b63\u5728\u751f\u6210\u9884\u89c8...")
        for i in range(preview_count):
            row = rows[i]
            text_items = build_text_items(row)
            img = generate_one(bg, text_items, img_width, font_color, font_path)
            fname = build_filename(row)
            issues = check_image_quality(img, text_items, img_width, qr_box, font_path)
            preview_imgs.append((img.copy(), fname))
            all_issues.append((fname, issues))
            progress.progress((i + 1) / preview_count,
                              text=f"\u9884\u89c8 [{i+1}/{preview_count}]")
        progress.progress(1.0, text=f"\u9884\u89c8\u5b8c\u6210! \u5171 {preview_count} \u5f20")
        st.session_state["preview_imgs"] = preview_imgs
        st.session_state["preview_issues"] = all_issues

    if "preview_imgs" in st.session_state and st.session_state["preview_imgs"]:
        preview_imgs = st.session_state["preview_imgs"]
        for i in range(0, len(preview_imgs), 3):
            chunk = preview_imgs[i:i+3]
            cols = st.columns(len(chunk))
            for col, (img, caption) in zip(cols, chunk):
                with col:
                    st.image(img, caption=caption, use_container_width=True)

        st.markdown("---")
        with st.expander("\U0001f50d \u70b9\u51fb\u653e\u5927\u67e5\u770b\u7ec6\u8282", expanded=False):
            selected_preview = st.selectbox(
                "\u9009\u62e9\u56fe\u7247",
                range(len(preview_imgs)),
                format_func=lambda i: preview_imgs[i][1],
            )
            st.image(preview_imgs[selected_preview][0],
                     caption=preview_imgs[selected_preview][1],
                     use_container_width=False)

        st.markdown("---")
        if "preview_gallery_confirmed" not in st.session_state:
            st.session_state["preview_gallery_confirmed"] = False

        op_col1, op_col2 = st.columns(2)
        with op_col1:
            if st.button("有错误点击重新生成预览", use_container_width=True, key="btn_regen_preview_gallery"):
                st.session_state["preview_gallery_confirmed"] = False
                for k in ["preview_imgs", "preview_issues", "all_img_data", "check_done"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with op_col2:
            if st.button("无问题确认效果", type="primary", use_container_width=True, key="btn_confirm_preview_gallery"):
                st.session_state["preview_gallery_confirmed"] = True
                st.success("已确认预览效果无问题，可以继续生成全部。")

        # ── generate all ──
        st.markdown("---")
        st.markdown("### \u751f\u6210\u5168\u90e8")
        if not st.session_state.get("preview_gallery_confirmed", False):
            st.info("请先点击“无问题确认效果”，再进行全部生成。")
        if st.button(f"\u751f\u6210\u5168\u90e8 {total} \u5f20",
                     type="primary", use_container_width=True,
                     disabled=not st.session_state.get("preview_gallery_confirmed", False)):
            progress2 = st.progress(0, text="\u6b63\u5728\u751f\u6210...")
            all_img_data = []
            for i, row in enumerate(rows):
                fname = build_filename(row)
                img = generate_one(bg, build_text_items(row), img_width, font_color, font_path)
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                all_img_data.append((f"{fname}.png", img_buf.getvalue()))
                progress2.progress((i + 1) / total,
                                   text=f"\u6b63\u5728\u751f\u6210 [{i+1}/{total}] {fname}")

            progress2.progress(1.0, text=f"\u5168\u90e8\u751f\u6210\u5b8c\u6210! \u5171 {total} \u5f20")
            st.session_state["all_img_data"] = all_img_data
            st.session_state["check_done"] = False

    # ── quality check after generation ──
    if "all_img_data" in st.session_state and st.session_state["all_img_data"]:
        all_img_data = st.session_state["all_img_data"]

        if not st.session_state.get("check_done", False):
            st.markdown("---")
            st.markdown("### \u8d28\u91cf\u68c0\u67e5")
            if st.button("\U0001f50d \u4e00\u952e\u68c0\u67e5\u6240\u6709\u56fe\u7247", use_container_width=True):
                check_count = min(10, len(all_img_data))
                all_check_issues = []
                check_progress = st.progress(0, text="\u6b63\u5728\u68c0\u67e5...")
                for i in range(check_count):
                    fname = all_img_data[i][0]
                    img = Image.open(io.BytesIO(all_img_data[i][1])).convert("RGBA")
                    row = rows[i]
                    text_items = build_text_items(row)
                    issues = check_image_quality(img, text_items, img_width, qr_box, font_path)
                    all_check_issues.append((fname, issues))
                    check_progress.progress((i + 1) / check_count,
                                            text=f"\u68c0\u67e5 [{i+1}/{check_count}]")
                check_progress.progress(1.0, text="\u68c0\u67e5\u5b8c\u6210!")

                has_errors = False
                has_warnings = False
                for fname, issues in all_check_issues:
                    for level, msg in issues:
                        if level == "error":
                            st.error(f"**{fname}**: {msg}")
                            has_errors = True
                        elif level == "warning":
                            st.warning(f"**{fname}**: {msg}")
                            has_warnings = True

                if not has_errors and not has_warnings:
                    st.success("\u2705 \u5168\u90e8\u68c0\u67e5\u901a\u8fc7! \u5b57\u4f53\u5c45\u4e2d / \u95f4\u8ddd\u6b63\u5e38 / \u4e8c\u7ef4\u7801\u53ef\u8bc6\u522b")
                    st.session_state["check_done"] = True
                elif not has_errors:
                    st.info("\u68c0\u67e5\u5b8c\u6210, \u6709\u8f7b\u5fae\u8b66\u544a\u4f46\u4e0d\u5f71\u54cd\u4f7f\u7528")
                    st.session_state["check_done"] = True
                else:
                    if st.button("\u267b\ufe0f \u68c0\u6d4b\u5230\u9519\u8bef, \u70b9\u51fb\u91cd\u65b0\u751f\u6210", type="primary", use_container_width=True):
                        for k in ["preview_imgs", "preview_issues", "all_img_data", "check_done"]:
                            st.session_state.pop(k, None)
                        st.rerun()

        if st.session_state.get("check_done", False):
            st.markdown("---")
            total_gen = len(all_img_data)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, data in all_img_data:
                    zf.writestr(filename, data)

            st.download_button(
                label=f"\u2705 \u68c0\u67e5\u901a\u8fc7, \u4e0b\u8f7d\u5168\u90e8 ({total_gen} \u5f20 ZIP)",
                data=zip_buf.getvalue(),
                file_name="\u9080\u8bf7\u51fd\u6279\u91cf\u751f\u6210.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
else:
    st.info("\u8bf7\u5148\u4e0a\u4f20\u6a21\u677f\u6587\u4ef6\u548c\u540d\u5355\u6587\u4ef6")

st.markdown("---")
with st.expander("\u57fa\u7840\u95ee\u9898\u89e3\u8bf4", expanded=False):
    st.markdown(
        "- \u82e5\u51fa\u73b0 `TypeError: Failed to fetch dynamically imported module`\uff0c\u901a\u5e38\u662f\u6d4f\u89c8\u5668\u7f13\u5b58\u6216\u7f51\u7edc\u62e6\u622a\u3002\n"
        "- \u8bf7\u6309\u987a\u5e8f\u5904\u7406\uff1a\u5173\u95ed\u9875\u9762\u91cd\u5f00 \u2192 \u5f3a\u5236\u5237\u65b0\uff08Windows: `Ctrl+Shift+R`\uff0cmacOS: `Cmd+Shift+R`\uff09\u2192 \u65e0\u75d5\u7a97\u53e3\u91cd\u8bd5\u3002\n"
        "- \u4ecd\u65e0\u6cd5\u8bbf\u95ee\u65f6\uff0c\u8bf7\u6e05\u9664 `streamlit.app` \u7ad9\u70b9\u6570\u636e\u540e\u518d\u8bd5\u3002\n"
        "- \u672c\u5de5\u5177\u4e3a Streamlit Cloud \u516c\u7f51\u90e8\u7f72\uff0c\u4e0d\u540c\u7f51\u7edc/\u5728\u5bb6\u5747\u53ef\u4f7f\u7528\uff1b\u82e5\u4ec5\u67d0\u4e9b\u7f51\u7edc\u5931\u8d25\uff0c\u8bf7\u8054\u7cfb IT \u653e\u884c `*.streamlit.app` HTTPS \u8bbf\u95ee\u3002"
    )
