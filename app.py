import io
import os
import re
import wave
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# لیست مدل‌های TTS به ترتیب اولویت فراخوانی
TTS_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

VOICES = ["Fenrir", "Kore", "Charon", "Puck", "Orus", "Zephyr", "Leda", "Aoede"]
GEMINI_DEFAULT_RATE = 24000  # خروجی PCM جمینای ۲۴ کیلوهرتز، ۱۶ بیت مونو است
SILENCE_SECONDS = 1.0

st.set_page_config(page_title="تیزر رادیویی فارسی", page_icon="🎙️", layout="centered")

st.markdown(
    """
    <style>
    textarea { direction: rtl; text-align: right; font-size: 1.05rem; }
    .stMarkdown, h1, h2, h3 { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_api_key(sidebar_key: str) -> str:
    if sidebar_key.strip():
        return sidebar_key.strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

def _parse_rate(mime_type: str) -> int:
    m = re.search(r"rate=(\d+)", mime_type or "")
    return int(m.group(1)) if m else GEMINI_DEFAULT_RATE

def _extract_audio(response):
    """اولین inline_data صوتی را برمی‌گرداند: (bytes, mime_type)"""
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data, (inline.mime_type or "")
    return None, ""

def _pcm_to_wav_bytes(pcm: bytes, rate: int, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()

def generate_announcer_wav(api_key: str, text: str, voice: str, models: list):
    """تولید صدای مجری فقط با متن فارسی بدون دستور انگلیسی"""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    errors = []
    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model, contents=text, config=config
                )
            except Exception as e:
                msg = str(e)
                errors.append(f"{model}: {msg[:200]}")
                low = msg.lower()
                if any(k in low for k in ("api key", "api_key", "permission", "401", "403")):
                    raise RuntimeError(f"مشکل کلید API یا دسترسی:\n{msg}") from e
                break

            data, mime = _extract_audio(response)
            if not data:
                errors.append(f"{model}: پاسخی حاوی داده صوتی نداشت (تلاش {attempt + 1})")
                continue

            if data[:4] == b"RIFF":
                return data, model
            return _pcm_to_wav_bytes(data, rate=_parse_rate(mime)), model

    raise RuntimeError("تولید صدا ناموفق بود:\n" + "\n".join(errors))

def _bytes_to_float(frames: bytes, width: int) -> np.ndarray:
    if width == 1:
        return (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.int32).reshape(-1, 3)
        v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        return v.astype(np.float32) / 8388608.0
    if width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"عمق بیت پشتیبانی‌نشده: {width * 8}-bit")

def read_wav(data: bytes):
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            ch, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"فایل WAV خوانده نشد. لطفاً WAV استاندارد (PCM) آپلود کنید. جزئیات: {e}") from e

    frames = frames[: len(frames) - (len(frames) % (width * ch))]
    arr = _bytes_to_float(frames, width).reshape(-1, ch)
    return arr, rate

def _match_channels(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[1] == target:
        return arr
    mono = arr.mean(axis=1, keepdims=True)
    return np.repeat(mono, target, axis=1)

def _resample(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(arr) == 0:
        return arr
    n_out = int(round(len(arr) * dst / src))
    x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.stack([np.interp(x_new, x_old, arr[:, c]) for c in range(arr.shape[1])], axis=1).astype(np.float32)

def merge_wavs(announcer_wav: bytes, user_wav: bytes, silence_sec: float = SILENCE_SECONDS) -> bytes:
    ann, ann_rate = read_wav(announcer_wav)
    usr, usr_rate = read_wav(user_wav)
    target_rate = usr_rate
    target_ch = usr.shape[1]

    ann = _resample(_match_channels(ann, target_ch), ann_rate, target_rate)
    silence = np.zeros((int(target_rate * silence_sec), target_ch), dtype=np.float32)
    combined = np.concatenate([ann, silence, usr], axis=0)
    pcm16 = (np.clip(combined, -1.0, 1.0) * 32767.0).astype("<i2")
    return _pcm_to_wav_bytes(pcm16.tobytes(), rate=target_rate, channels=target_ch, width=2)

# UI اصلی برنامه
st.title("🎙️ ساخت تیزر صوتی رادیویی")
st.caption("متن معرفی مجری → صدای Gemini → الصاق به ابتدای دکلمه شما")

with st.sidebar:
    st.header("تنظیمات")
    key_input = st.text_input("Gemini API Key", type="password", help="اگر در Secrets تنظیم شده خالی بگذارید.")
    voice = st.selectbox("صدای مجری", VOICES, index=0)
    model_choice = st.selectbox("مدل", ["خودکار (با fallback)"] + TTS_MODELS)
    custom_model = st.text_input("نام مدل دلخواه (اختیاری)")

intro_text = st.text_area(
    "متن معرفی مجری (فارسی)",
    height=150,
    placeholder="مثلاً: شنوندگان عزیز در ادامه برنامه‌ای را می‌شنوید..."
)

uploaded = st.file_uploader("فایل دکلمه (WAV)", type=["wav"])

if uploaded:
    st.audio(uploaded.getvalue(), format="audio/wav")

if st.button("🚀 تولید تیزر", type="primary", use_container_width=True):
    api_key = get_api_key(key_input)
    text = intro_text.strip()

    if not api_key:
        st.error("کلید API وارد نشده است.")
    elif not text:
        st.error("متن مجری را وارد کنید.")
    elif not uploaded:
        st.error("فایل WAV دکلمه را آپلود کنید.")
    else:
        if custom_model.strip():
            models = [custom_model.strip()]
        elif model_choice.startswith("خودکار"):
            models = TTS_MODELS
        else:
            models = [model_choice]

        try:
            with st.spinner("در حال تولید صدای مجری..."):
                ann_wav, used_model = generate_announcer_wav(api_key, text, voice, models)

            with st.spinner("در حال ترکیب فایل‌ها..."):
                final_wav = merge_wavs(ann_wav, uploaded.getvalue())

            st.session_state["result"] = {
                "announcer": ann_wav,
                "final": final_wav,
                "model": used_model,
            }
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(str(e))

result = st.session_state.get("result")
if result:
    st.success(f"تیزر آماده شد! (مدل: {result['model']})")
    st.subheader("پیش‌نمایش نهایی")
    st.audio(result["final"], format="audio/wav")
    st.download_button(
        "⬇️ دانلود تیزر (WAV)",
        data=result["final"],
        file_name="teaser.wav",
        mime="audio/wav",
        use_container_width=True,
    )
    with st.expander("فقط صدای مجری"):
        st.audio(result["announcer"], format="audio/wav")
