#!/usr/bin/env python3
"""批量邀请函生成工具 — Streamlit Web 应用 v3 (Cloud)"""

import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from psd_tools import PSDImage

APP_DIR = Path(__file__).parent
FONTS_DIR = APP_DIR / "fonts"

st.set_page_config(page_title="批量邀请函生成", page_icon="📨", layout="centered")
st.title("批量邀请函生成工具")

# ── password gate ────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("请输入访问密码", type="password")
    if pwd == "950621":
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:
        st.error("密码错误")
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


def replace_qr(background, qr_image, qr_box, corner_radius=5):
    tw = qr_box[2] - qr_box[0]
    th = qr_box[3] - qr_box[1]
    qr_resized = qr_image.resize((tw, th), Image.LANCZOS)
    mask = rounded_corner_mask((tw, th), corner_radius)
    qr_resized.putalpha(mask)
    background.paste(qr_resized, (qr_box[0], qr_box[1]), qr_resized)
    return background


def get_font_info(psd):
    for l in psd.descendants():
        if l.kind == "type":
            ss = l.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
            size = ss.get("FontSize", 51)
            color_vals = ss.get("FillColor", {}).get("Values", [1.0, 1.0, 1.0, 1.0])
            rgba = tuple(int(v * 255) for v in color_vals)
            return size, rgba
    return 51, (255, 255, 255, 255)


def get_text_layer_positions(psd):
    positions = {}
    for l in psd.descendants():
        if l.kind == "type":
            cx = (l.left + l.right) // 2
            positions[l.name] = (cx, l.top)
    return positions


def draw_centered_text(draw, font, text, y, img_width, color):
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    x = (img_width - text_w) // 2
    draw.text((x, y), text, font=font, fill=color)


def generate_one(background, font, company, name, company_y, name_y, img_width, color):
    img = background.copy()
    draw = ImageDraw.Draw(img)
    draw_centered_text(draw, font, company, company_y, img_width, color)
    draw_centered_text(draw, font, name, name_y, img_width, color)
    return img


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
        "1. 上传模板文件",
        type=ALL_TEMPLATE_TYPES,
        help="支持 PSD / PSB / PNG / JPG / TIFF / BMP / WebP / PDF / EPS / AI",
    )
with col2:
    list_file = st.file_uploader(
        "2. 上传名单",
        type=LIST_EXTENSIONS,
        help="支持 CSV / Excel (.xlsx) / 旧版 Excel (.xls)",
    )

qr_file = st.file_uploader("3. 上传替换二维码（可选）", type=["png", "jpg", "jpeg", "webp"])

if template_file and list_file:
    suffix = file_suffix(template_file)
    is_psd = suffix in PSD_EXTENSIONS

    with st.spinner("正在解析模板..."):
        if is_psd:
            psd = load_psd(template_file)
            text_layers = get_text_layers(psd)
            layer_names = [l.name for l in text_layers]
            positions = get_text_layer_positions(psd)
            font_size, font_color = get_font_info(psd)
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

    # ── field mapping ──
    st.markdown("### 字段映射")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        company_field = st.selectbox("公司名对应字段", fields, index=0)
    with mcol2:
        name_idx = min(1, len(fields) - 1)
        name_field = st.selectbox("人名对应字段", fields, index=name_idx)

    if is_psd and layer_names:
        lcol1, lcol2 = st.columns(2)
        with lcol1:
            company_layer = st.selectbox("公司名对应 PSD 图层", layer_names,
                                         index=max(0, len(layer_names) - 1))
        with lcol2:
            name_layer = st.selectbox("人名对应 PSD 图层", layer_names, index=0)
        company_y = positions[company_layer][1]
        name_y = positions[name_layer][1]
    else:
        st.markdown("**文字位置设置**（图片模板需手动指定 Y 坐标）")
        pcol1, pcol2 = st.columns(2)
        default_company_y = int(img_height * 0.45)
        default_name_y = int(img_height * 0.48)
        with pcol1:
            company_y = st.number_input("公司名 Y 坐标", 0, img_height, default_company_y)
        with pcol2:
            name_y = st.number_input("人名 Y 坐标", 0, img_height, default_name_y)

    # ── font selection ──
    st.markdown("### 字体选择")
    all_fonts = scan_fonts()
    font_names = list(all_fonts.keys())
    default_idx = 0
    for i, name in enumerate(font_names):
        if "OPPO Sans 4.0" in name:
            default_idx = i
            break
    selected_font = st.selectbox(
        "选择字体",
        font_names,
        index=default_idx,
        help=f"已扫描到 {len(font_names)} 个可用字体",
    )
    font_path = all_fonts[selected_font]

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

    font = ImageFont.truetype(font_path, int(font_size))

    # ── preview ──
    st.markdown("### 预览对比")
    first = rows[0]
    preview = generate_one(bg, font, first[company_field], first[name_field],
                           company_y, name_y, img_width, font_color)

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.image(original_img, caption="原始模板", use_container_width=True)
    with pcol2:
        st.image(preview, caption=f"替换效果：{first[company_field]} — {first[name_field]}",
                 use_container_width=True)

    # ── batch generate ──
    st.markdown("### 批量生成")
    if st.button("开始生成全部", type="primary", use_container_width=True):
        progress = st.progress(0, text="准备中...")
        zip_buf = io.BytesIO()
        total = len(rows)
        preview_imgs = []

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, row in enumerate(rows):
                company = row[company_field]
                name = row[name_field]
                img = generate_one(bg, font, company, name,
                                   company_y, name_y, img_width, font_color)
                if i < 3:
                    preview_imgs.append((img.copy(), f"{company}_{name}"))

                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"{company}_{name}.png", img_buf.getvalue())
                progress.progress((i + 1) / total, text=f"正在生成 [{i+1}/{total}] {company}_{name}")

        progress.progress(1.0, text=f"全部完成！共 {total} 张")
        st.balloons()

        if preview_imgs:
            st.markdown("#### 生成效果预览")
            cols = st.columns(len(preview_imgs))
            for col, (img, caption) in zip(cols, preview_imgs):
                with col:
                    st.image(img, caption=caption, use_container_width=True)

        st.download_button(
            label=f"下载全部 ({total} 张 ZIP)",
            data=zip_buf.getvalue(),
            file_name="邀请函批量生成.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
else:
    st.info("请先上传模板文件和名单文件")
