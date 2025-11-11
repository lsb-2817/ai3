# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1xnLTNeL0MueJFgrwu7yWW2IgTf6VT6yH")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    
    labels[0]: {
       "texts": ["밀은 맛있다"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=GGDl66NnOoY"]
    },


labels[0]: {
       "texts": ["밀은 맛있다"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=GGDl66NnOoY"]
    },

labels[0]: {
       "texts": ["밀은 맛있다"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=GGDl66NnOoY"]
    },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
