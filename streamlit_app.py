import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlaucomaNet · Eye Disease Detector",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .main-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    .result-glaucoma {
        background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(185,28,28,0.15));
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-normal {
        background: linear-gradient(135deg, rgba(34,197,94,0.25), rgba(21,128,61,0.15));
        border: 1px solid rgba(34,197,94,0.5);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-pill {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 999px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        color: #e0e0e0;
        margin: 0.2rem;
    }
    .conf-track {
        background: rgba(255,255,255,0.1);
        border-radius: 999px;
        height: 12px;
        width: 100%;
        margin-top: 0.5rem;
    }
    h1, h2, h3 { color: #ffffff !important; }
    p, li, label { color: #cbd5e1 !important; }
    .stMarkdown p { color: #cbd5e1; }
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255,255,255,0.04) !important;
        border: 2px dashed rgba(148,163,184,0.4) !important;
        border-radius: 14px !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ GlaucomaNet")
    st.markdown("**AI-powered glaucoma screening** from retinal fundus images.")
    st.divider()

    st.markdown("### 📂 Model")
    model_path = st.text_input(
        "Model file path (.h5 / .keras)",
        value="combine_cnn.h5",
        help="Path to your saved Keras model file",
    )

    st.divider()
    st.markdown("### ⚙️ Settings")
    img_size = st.select_slider(
        "Input resolution",
        options=[128, 192, 224, 256],
        value=256,
        help="Must match the size used during training (default 256×256)",
    )
    show_debug = st.checkbox("Show debug info", value=False)

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
Trained on three public fundus datasets:
- **DRISHTI** – 101 images (IIIT Hyderabad)
- **RIM-ONE DL** – 485 images (3 Spanish hospitals)
- **ACRIMA** – 705 images (FISABIO Valencia)

Architecture: Custom CNN · Input: 256×256 RGB · Output: Glaucoma / Normal
""")
    st.caption("For research purposes only. Not a medical device.")


# ─── Model loader (cached) ───────────────────────────────────────────────────
'''@st.cache_resource(show_spinner="Loading model weights…")
def load_cnn_model(path: str):
    """Load the Keras model. Returns (model, error_string)."""
    try:
        from tensorflow.keras.models import load_model  # type: ignore
        if not os.path.exists(path):
            return None, f"File not found: `{path}`"
        model = load_model(path)
        return model, None
    except ImportError:
        return None, "TensorFlow is not installed. Run `pip install tensorflow`."
    except Exception as e:
        return None, str(e)'''


# ─── Prediction helper ───────────────────────────────────────────────────────
def predict(model, pil_image: Image.Image, target_size: int):
    """Return (label, confidence_glaucoma, confidence_normal)."""
    from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
    img = pil_image.convert("RGB").resize((target_size, target_size))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]   # shape: (2,)
    label_idx = int(np.argmax(probs))
    # Class order from ImageDataGenerator (alphabetical): glaucoma=0, normal=1
    labels = ["Glaucoma", "Normal"]
    return labels[label_idx], float(probs[0]), float(probs[1])


# ─── Header ──────────────────────────────────────────────────────────────────
col_icon, col_title = st.columns([1, 10])
with col_icon:
    st.markdown("<div style='font-size:3.5rem;margin-top:0.3rem'>👁️</div>",
                unsafe_allow_html=True)
with col_title:
    st.markdown("# GlaucomaNet")
    st.markdown("<p style='margin-top:-0.8rem;color:#94a3b8;font-size:1.05rem'>"
                "Retinal Fundus Image Classifier · CNN-based Glaucoma Detection</p>",
                unsafe_allow_html=True)

st.divider()

# ─── Load model ──────────────────────────────────────────────────────────────
model, model_err = load_cnn_model(model_path)

if model_err:
    st.warning(f"⚠️ Model not loaded — {model_err}")
    st.info(
        "Place your `combine_cnn.h5` file in the same directory as this script, "
        "or adjust the path in the sidebar. The UI is fully functional once the model is loaded."
    )
    model = None
else:
    st.success("✅ Model loaded successfully")
    if show_debug:
        st.write(f"**Input shape:** {model.input_shape} | "
                 f"**Parameters:** {model.count_params():,}")

st.markdown("")

# ─── Upload section ──────────────────────────────────────────────────────────
st.markdown("### 🔬 Upload Fundus Image(s)")
st.markdown(
    "<p style='color:#94a3b8'>Upload one or more retinal fundus images (JPG / PNG / BMP). "
    "The model will classify each as <b>Glaucoma</b> or <b>Normal</b>.</p>",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Drop images here",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""
    <div class="main-card" style="text-align:center;padding:3rem">
        <div style="font-size:3rem">🏥</div>
        <h3>No images uploaded yet</h3>
        <p>Drag and drop retinal fundus photographs above to begin screening.</p>
        <span class="metric-pill">Accepted formats: JPG · PNG · BMP</span>
        <span class="metric-pill">Optimal resolution: 256×256 px</span>
    </div>
    """, unsafe_allow_html=True)
else:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        img_col, result_col = st.columns([1, 1], gap="large")

        pil_img = Image.open(io.BytesIO(uploaded_file.read()))

        with img_col:
            st.markdown("<div class='main-card'>", unsafe_allow_html=True)
            st.image(pil_img, caption=uploaded_file.name, use_container_width=True)
            w, h = pil_img.size
            st.markdown(
                f"<span class='metric-pill'>📐 {w}×{h} px</span>"
                f"<span class='metric-pill'>🗂 {uploaded_file.type}</span>"
                f"<span class='metric-pill'>💾 {uploaded_file.size/1024:.1f} KB</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col:
            if model is None:
                st.markdown("""
                <div class="main-card" style="text-align:center;padding:2rem">
                    <div style="font-size:2.5rem">⚠️</div>
                    <h3>Model not loaded</h3>
                    <p>Please provide a valid model path in the sidebar.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Analysing image…"):
                    label, conf_g, conf_n = predict(model, pil_img, img_size)

                is_glaucoma = label == "Glaucoma"
                card_class = "result-glaucoma" if is_glaucoma else "result-normal"
                emoji = "🔴" if is_glaucoma else "🟢"
                conf_pct = conf_g * 100 if is_glaucoma else conf_n * 100
                bar_color = "#ef4444" if is_glaucoma else "#22c55e"

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-size:3rem">{emoji}</div>
                    <h2 style="margin:0.4rem 0 0.2rem">{label}</h2>
                    <p style="font-size:0.9rem;margin:0;color:#e0e0e0">
                        Confidence: <strong>{conf_pct:.1f}%</strong>
                    </p>
                    <div class="conf-track">
                        <div style="background:{bar_color};height:100%;
                                    border-radius:999px;width:{conf_pct}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='main-card'>", unsafe_allow_html=True)
                st.markdown("**Probability breakdown**")
                p1, p2 = st.columns(2)
                p1.metric("🔴 Glaucoma", f"{conf_g*100:.1f}%")
                p2.metric("🟢 Normal",   f"{conf_n*100:.1f}%")

                if show_debug:
                    st.markdown("**Raw softmax output**")
                    st.code(f"[glaucoma={conf_g:.6f}, normal={conf_n:.6f}]")
                st.markdown("</div>", unsafe_allow_html=True)

                if is_glaucoma:
                    st.error(
                        "⚠️ **Possible glaucoma detected.** "
                        "Please refer to an ophthalmologist for a full clinical evaluation."
                    )
                else:
                    st.success(
                        "✅ **No signs of glaucoma detected.** "
                        "Continue routine eye check-ups as recommended."
                    )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem'>"
    "GlaucomaNet · CNN trained on DRISHTI + RIM-ONE + ACRIMA datasets · "
    "For research and educational purposes only — not a substitute for professional medical diagnosis."
    "</p>",
    unsafe_allow_html=True,
)
