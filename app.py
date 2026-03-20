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
from psd_tools import PSDImage

APP_DIR = Path(__file__).parent
FONTS_DIR = APP_DIR / "fonts"

st.set_page_config(page_title="批量邀请函生成", page_icon="📨", layout="centered")
st.title("批量邀请函生成工具")

# Streamlit 內建上傳元件預設是英文，這裡用樣式覆蓋成中文提示。
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] > div {
        display: none;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "將檔案拖曳到這裡";
        display: block;
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "每個檔案上限 2GB";
        display: block;
        font-size: 1rem;
        color: rgba(250, 250, 250, 0.72);
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "選擇檔案";
        font-size: 1.25rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.expander("\u540c\u4e8b\u7aef\u7ea2\u6846\u62a5\u9519\u6392\u67e5", expanded=False):
    st.markdown(
        "- `TypeError: Failed to fetch dynamically imported module` \u901a\u5e38\u662f\u6d4f\u89c8\u5668\u7f13\u5b58\u6216\u7f51\u7edc\u963b\u62e6\u5bfc\u81f4\u3002\n"
        "- \u8bf7\u4f9d\u6b21\u5c1d\u8bd5\uff1a\u5173\u95ed\u9875\u9762\u91cd\u5f00 \u2192 \u5f3a\u5236\u5237\u65b0 (Win: `Ctrl+Shift+R`, macOS: `Cmd+Shift+R`) \u2192 \u65e0\u75d5\u6a21\u5f0f\u3002\n"
        "- \u4ecd\u5931\u8d25\u65f6\uff0c\u6e05\u9664 `streamlit.app` \u7ad9\u70b9\u6570\u636e\u540e\u91cd\u8bd5\u3002\n"
        "- \u82e5\u516c\u53f8\u7f51\u7edc\u4e0d\u884c\u3001\u624b\u673a\u7f51\u7edc\u53ef\u4ee5\uff0c\u8bf7 IT \u653e\u884c `*.streamlit.app` HTTPS \u8bbf\u95ee\u3002"
    )
st.caption(
    "\u516c\u7f51\u8bf4\u660e\uff1aStreamlit Cloud \u90e8\u7f72\u6210\u529f\u540e\uff0c\u4e0d\u540c\u7f51\u7edc\u548c\u5728\u5bb6\u4e00\u822c\u90fd\u53ef\u6253\u5f00\u4f7f\u7528\uff08\u9664\u975e\u5f53\u5730/\u516c\u53f8\u7f51\u7edc\u5bf9\u8be5\u57df\u540d\u6709\u9650\u5236\uff09\u3002"
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


def replace_qr(background, qr_image, qr_box, corner_radius=3):
    tw = qr_box[2] - qr_box[0]
    th = qr_box[3] - qr_box[1]
    qr_resized = qr_image.convert("RGBA").resize((tw, th), Image.LANCZOS)
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
    """Return {layer_name: (center_y, calibrated_font_size, psd_width)}."""
    positions = {}
    for l in psd.descendants():
        if l.kind == "type":
            cy = (l.top + l.bottom) // 2
            ss = l.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
            raw_size = ss.get("FontSize", 51)
            cal_size = int(raw_size)
            if font_path and l.height > 0 and len(l.text) >= 1:
                cal_size = calibrate_font_size(font_path, l.text, l.height, raw_size)
            positions[l.name] = (cy, cal_size, l.width)
    return positions


def draw_centered_text(draw, font, text, center_y, img_width, color):
    """Draw text horizontally and vertically centered using standard draw.text."""
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    x = (img_width - text_w) // 2
    y = center_y - (bbox[1] + bbox[3]) // 2
    draw.text((x, y), text, font=font, fill=color)


def generate_one(background, text_items, img_width, color, font_path):
    """text_items: list of (text_str, center_y, font_size)."""
    img = background.copy()
    draw = ImageDraw.Draw(img)
    for text, cy, fsize in text_items:
        if text:
            f = ImageFont.truetype(font_path, fsize)
            draw_centered_text(draw, f, text, cy, img_width, color)
    return img


def check_image_quality(img, text_items, img_width, qr_box, font_path):
    """Quality checks: text bounds, spacing vs PSD width, QR readability."""
    issues = []

    for text, center_y, fsize in text_items:
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

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader(
        "1. \u4e0a\u4f20\u6a21\u677f\u6587\u4ef6",
        type=ALL_TEMPLATE_TYPES,
        help="PSD / PSB / PNG / JPG / TIFF / BMP / WebP / PDF / EPS / AI",
    )
with col2:
    list_file = st.file_uploader(
        "2. \u4e0a\u4f20\u540d\u5355",
        type=LIST_EXTENSIONS,
        help="CSV / Excel (.xlsx) / Excel (.xls)",
    )

qr_file = st.file_uploader(
    "3. \u4e0a\u4f20\u66ff\u6362\u4e8c\u7ef4\u7801 (\u53ef\u9009)",
    type=["png", "jpg", "jpeg", "webp"],
)

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
    st.markdown("### \u5b57\u6bb5\u6620\u5c04")

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
    name_field = None
    name_y = 0
    name_fsize = font_size
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
        bg = replace_qr(bg, qr_img, qr_box)

    def build_text_items(row):
        items = []
        if enable_company and company_field:
            items.append((row[company_field], company_y, company_fsize))
        if enable_name and name_field:
            items.append((row[name_field], name_y, name_fsize))
        return items

    def build_filename(row):
        parts = []
        if enable_company and company_field:
            parts.append(row[company_field])
        if enable_name and name_field:
            parts.append(row[name_field])
        return "_".join(parts) if parts else f"row"

    # ── preview: original vs first ──
    st.markdown("### \u9884\u89c8\u5bf9\u6bd4")
    first = rows[0]
    preview = generate_one(bg, build_text_items(first), img_width, font_color, font_path)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.image(original_img, caption="\u539f\u59cb\u6a21\u677f", use_container_width=True)
    with pcol2:
        st.image(preview, caption=f"\u66ff\u6362\u6548\u679c: {build_filename(first)}",
                 use_container_width=True)

    st.markdown("---")
    st.caption("\u8bf7\u4ed4\u7ec6\u5bf9\u6bd4\u4e0a\u65b9\u4e24\u5f20\u56fe\uff0c\u786e\u8ba4\u5b57\u4f53\u5927\u5c0f\u3001\u4f4d\u7f6e\u3001\u95f4\u8ddd\u3001\u4e8c\u7ef4\u7801\u662f\u5426\u4e0e\u539f\u56fe\u4e00\u81f4\u3002")

    if "preview_confirmed" not in st.session_state:
        st.session_state.preview_confirmed = False

    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        if st.button("\u2705 \u6548\u679c\u6b63\u786e\uff0c\u7ee7\u7eed\u4e0b\u4e00\u6b65", type="primary", use_container_width=True):
            st.session_state.preview_confirmed = True
            st.rerun()
    with confirm_col2:
        report = st.text_input("\u274c \u53d1\u73b0\u9519\u8bef\uff1f\u8bf7\u63cf\u8ff0\u95ee\u9898", placeholder="\u4f8b\u5982: \u5b57\u4f53\u504f\u5c0f / \u4f4d\u7f6e\u504f\u79fb / \u95f4\u8ddd\u4e0d\u5bf9...")
        if report:
            st.warning(f"\u4f60\u53cd\u9988\u7684\u95ee\u9898: \u300c{report}\u300d")
            st.info("\u8bf7\u8c03\u6574\u4e0a\u65b9\u7684\u5b57\u6bb5\u6620\u5c04\u6216\u5b57\u4f53\u8bbe\u7f6e\u540e\uff0c\u9875\u9762\u4f1a\u81ea\u52a8\u66f4\u65b0\u9884\u89c8\u3002\u786e\u8ba4\u65e0\u8bef\u540e\u518d\u70b9\u201c\u6548\u679c\u6b63\u786e\u201d\u3002")
            st.session_state.preview_confirmed = False
            st.stop()

    if not st.session_state.preview_confirmed:
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
        if st.button("\u267b\ufe0f \u91cd\u65b0\u751f\u6210\u9884\u89c8", use_container_width=True):
            for k in ["preview_imgs", "preview_issues", "all_img_data", "check_done"]:
                st.session_state.pop(k, None)
            st.rerun()

        # ── generate all ──
        st.markdown("---")
        st.markdown("### \u751f\u6210\u5168\u90e8")
        if st.button(f"\u751f\u6210\u5168\u90e8 {total} \u5f20",
                     type="primary", use_container_width=True):
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
