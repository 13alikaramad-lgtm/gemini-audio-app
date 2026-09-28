import io
import os
import re
import wave

import numpy as np
import streamlit as st
from google import genai
from google.genai import errors, types

# ---------------------------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------------------------
# توصیه: این مقدار را خالی بگذارید و کلید را در Secrets ذخیره کنید.
HARDCODED_API_KEY = ""

TTS_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
]
VOICES = ["Fenrir", "Kore", "Puck", "Charon", "Aoede", "Zephyr"]
SILENCE_SECONDS = 1.0
DEFAULT_GEMINI_RATE = 24000  # خروجی TTS جمینی: PCM 16 بیتی مونو


def get_api_key():
    """اولویت: کلید مستقیم ← st.secrets ← متغیرهای محیطی."""
    if HARDCODED_API_KEY and HARDCODED_API_KEY.strip():
        return HARDCODED_API_KEY.strip()
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return str(key).strip()
    except Exception:
        pass
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()


# ---------------------------------------------------------------------------
# تولید صدا با Gemini
# ---------------------------------------------------------------------------
def generate_tts(api_key, text, voice_name):
    """برمی‌گرداند: (pcm_bytes, sample_rate, model_used)"""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )

    failures = []
    for model in TTS_MODELS:
        try:
            # contents فقط متن فارسی کاربر است، بدون هیچ پرامپت اضافه
            response = client.models.generate_content(
                model=model, contents=text, config=config
            )
            part = response.candidates[0].content.parts[0]
            inline = part.inline_data
            if inline is None or not inline.data:
                failures.append(f"{model}: پاسخ صوتی خالی بود")
                continue

            rate = DEFAULT_GEMINI_RATE
            mime = getattr(inline, "mime_type", "") or ""
            m = re.search(r"rate=(\d+)", mime)
            if m:
                rate = int(m.group(1))
            return inline.data, rate, model

        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code == 404:
                failures.append(f"{model}: 404 (مدل یافت نشد)")
            else:
                failures.append(f"{model}: خطای {code} - {e}")
            continue
        except (IndexError, AttributeError, TypeError) as e:
            failures.append(f"{model}: پاسخ نامعتبر ({e})")
            continue

    raise RuntimeError(
        "هیچ‌کدام از مدل‌ها پاسخ ندادند:\n" + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# پردازش صوت با wave و numpy (بدون audioop)
# ---------------------------------------------------------------------------
def pcm16_to_float(pcm_bytes, channels=1):
    n = len(pcm_bytes) // 2
    data = np.frombuffer(pcm_bytes[: n * 2], dtype="<i2").astype(np.float32)
    usable = (len(data) // channels) * channels
    return data[:usable].reshape(-1, channels)


def read_wav_as_float(file_bytes):
    """خواندن WAV با هر عمق بیت PCM و تبدیل به float32 با شکل (n, ch)
    در بازه ۱۶ بیتی. برمی‌گرداند: (data, rate, channels)"""
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wf:
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except wave.Error as e:
        raise ValueError(
            f"فایل WAV پشتیبانی نمی‌شود (باید PCM باشد، نه Float): {e}"
        )

    if width == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        a = (a - 128.0) * 256.0
    elif width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif width == 3:
        b = np.frombuffer(raw[: (len(raw) // 3) * 3], dtype=np.uint8)
        b = b.reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = (v / 256.0).astype(np.float32)
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536.0
    else:
        raise ValueError(f"عمق بیت {width * 8} پشتیبانی نمی‌شود.")

    usable = (len(a) // ch) * ch
    return a[:usable].reshape(-1, ch), rate, ch


def resample(x, sr_in, sr_out):
    """ریسمپل خطی برای هر کانال."""
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = max(1, int(round(len(x) * sr_out / sr_in)))
    pos = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    xp = np.arange(len(x), dtype=np.float64)
    out = np.empty((n_out, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(pos, xp, x[:, c])
    return out


def match_channels(x, target_ch):
    cur = x.shape[1]
    if cur == target_ch:
        return x
    if cur == 1:
        return np.repeat(x, target_ch, axis=1)
    mono = x.mean(axis=1, keepdims=True)
    return np.repeat(mono, target_ch, axis=1)


def to_wav_bytes(data_float, rate, channels):
    pcm = np.clip(np.rint(data_float), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_final_audio(gemini_pcm, gemini_rate, user_wav_bytes):
    user_data, user_rate, user_ch = read_wav_as_float(user_wav_bytes)

    intro = pcm16_to_float(gemini_pcm, channels=1)
    intro = resample(intro, gemini_rate, user_rate)
    intro = match_channels(intro, user_ch)

    silence = np.zeros(
        (int(round(SILENCE_SECONDS * user_rate)), user_ch), dtype=np.float32
    )

    combined = np.concatenate([intro, silence, user_data], axis=0)
    return to_wav_bytes(combined, user_rate, user_ch), intro, user_rate, user_ch


# ---------------------------------------------------------------------------
# رابط کاربری
# ---------------------------------------------------------------------------
st.set_page_config(page_title="تیزر صوتی رادیویی", page_icon="🎙️")

st.markdown(
    """
    <style>
    .stTextArea textarea { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ تولید تیزر صوتی رادیویی")
st.caption("متن معرفی مجری را وارد کنید، دکلمه خود را آپلود کنید؛ خروجی نهایی آماده می‌شود.")

api_key = get_api_key()
if not api_key:
    st.error(
        "کلید API پیدا نشد. مقدار GEMINI_API_KEY را در بخش Secrets "
        "برنامه در Streamlit Cloud تنظیم کنید."
    )

voice = st.selectbox("صدای مجری", VOICES, index=0)
intro_text = st.text_area(
    "متن معرفی مجری (فارسی)", height=160, placeholder="متن را اینجا بنویسید..."
)
uploaded = st.file_uploader("فایل صوتی دکلمه (WAV)", type=["wav"])

if st.button("🚀 تولید تیزر", type="primary", disabled=not api_key):
    if not intro_text.strip():
        st.warning("لطفاً متن معرفی را وارد کنید.")
    elif uploaded is None:
        st.warning("لطفاً فایل WAV دکلمه را آپلود کنید.")
    else:
        try:
            with st.spinner("در حال تولید صدای مجری..."):
                pcm, rate, used_model = generate_tts(
                    api_key, intro_text.strip(), voice
                )
            with st.spinner("در حال ترکیب فایل‌ها..."):
                final_wav, _, out_rate, out_ch = build_final_audio(
                    pcm, rate, uploaded.getvalue()
                )
            st.session_state["result"] = {
                "wav": final_wav,
                "model": used_model,
                "rate": out_rate,
                "channels": out_ch,
            }
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"خطا: {e}")

result = st.session_state.get("result")
if result:
    st.success(
        f"آماده شد ✅ (مدل: {result['model']} | {result['rate']} Hz | "
        f"{result['channels']} کانال)"
    )
    st.audio(result["wav"], format="audio/wav")
    st.download_button(
        "⬇️ دانلود فایل نهایی (WAV)",
        data=result["wav"],
        file_name="radio_teaser.wav",
        mime="audio/wav",
    )
