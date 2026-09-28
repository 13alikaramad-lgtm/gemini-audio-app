import os
import streamlit as st
from google import genai
from google.genai import types

# تنظیمات اولیه صفحه
st.set_page_config(
    page_title="Gemini Audio Studio",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ استودیو و پرامپت‌ساز هوشمند صدای Gemini")

# نوار کناری تنظیمات
st.sidebar.header("🔑 تنظیمات اتصال")
api_key_input = st.sidebar.text_input("کلید API گوگل (GEMINI_API_KEY):", type="password")

st.sidebar.divider()
st.sidebar.header("🎙️ تنظیمات گوینده")
voice_option = st.sidebar.selectbox(
    "صدای پیش‌فرض:",
    options=["Puck", "Charon", "Kore", "Fenrir", "Aoede"],
    index=0
)

# تب‌های اصلی
tab_studio, tab_docs = st.tabs(["⚡ استودیوی ساخت صدا", "📖 راهنما"])

with tab_studio:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🛠️ تنظیمات سناریو")
        speaker_a_desc = st.text_input("توصیف گوینده اول:", value="A confident male with a warm, energetic voice")
        speaker_b_desc = st.text_input("توصیف گوینده دوم:", value="A calm female with a clear, steady voice")
        
        default_script = """Speaker A: |clears-throat| [cheerfully] سلام! نسخه جدید صوتی منتشر شد.
Speaker B: [warmly] سلام. |mhm| بله، کیفیتش فوق‌العاده‌ست!"""
        
        script_input = st.text_area("متن دیالوگ:", value=default_script, height=180)
        generate_btn = st.button("🚀 تولید و پردازش صوت", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎧 خروجی صوتی")
        if generate_btn:
            api_key = api_key_input or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                st.error("❌ لطفاً کلید API را وارد کنید.")
            else:
                full_prompt = f"""[Speakers Definition]:
- Speaker A: {speaker_a_desc}
- Speaker B: {speaker_b_desc}

[Script]:
{script_input}"""

                with st.spinner("درحال ساخت فایل صوتی..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=voice_option
                                        )
                                    )
                                )
                            )
                        )

                        audio_bytes = None
                        for part in response.candidates[0].content.parts:
                            if part.inline_data:
                                audio_bytes = part.inline_data.data
                                break

                        if audio_bytes:
                            st.success("✅ فایل صوتی ساخته شد!")
                            st.audio(audio_bytes, format="audio/wav")
                            st.download_button("📥 دانلود WAV", data=audio_bytes, file_name="gemini_output.wav", mime="audio/wav")
                        else:
                            st.warning("⚠️ داده صوتی یافت نشد.")
                    except Exception as e:
                        st.error(f"❌ خطا: {e}")

with tab_docs:
    st.markdown("### 📚 راهنمای برچسب‌ها\n* احساسی: `[cheerfully]`, `[warmly]`\n* افکت صوتی: `|clears-throat|`, `|sigh|`, `|mhm|`")
