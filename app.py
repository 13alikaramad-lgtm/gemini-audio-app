import streamlit as st
from google import genai
from google.genai import types
import wave
import io

st.set_page_config(page_title="استودیوی دکلمه و گویندگی", page_icon="🎙️", layout="centered")

st.title("🎙️ استودیوی گویندگی و ساخت تیزر رادیویی")
st.caption("تولید صدای مجری با هوش مصنوعی Gemini و ترکیب خودکار با دکلمه شما")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ تنظیمات API و گوینده")
    api_key = st.text_input("کلید API گوگل (Gemini API Key):", type="password")
    
    selected_voice = st.selectbox(
        "صدای مجری (Voice):",
        ["Puck", "Fenrir", "Kore", "Aoede", "Charon"],
        index=1,
        help="صدای Fenrir و Puck برای لحن‌های بم و مجری‌گری بسیار مناسب هستند."
    )
    
    st.markdown("---")
    st.markdown("💡 **راهنما:** کلید API خود را از [Google AI Studio](https://aistudio.google.com/) دریافت کنید.")

# Main Form
st.subheader("۱. متن معرفی مجری")
default_script = "اکنون از پدیده نوظهور در حوزه گویندگی و دکلمه اشعار، با ویژگی اجرای فاخر، جناب آقای علی کارآمد دعوت می‌کنیم برای اجرای یک دکلمه فاخر به جایگاه تشریف بیاورند."
script_text = st.text_area("متن دیالوگ مجری:", value=default_script, height=120)

st.subheader("۲. آپلود فایل صوتی دکلمه (استاد علی کارآمد)")
uploaded_declination = st.file_uploader("فایل صوتی دکلمه خود را انتخاب کنید (فرمت WAV توصیه می‌شود):", type=["wav", "mp3"])

system_instruction = st.text_input(
    "توصیف لحن و شخصیت مجری:",
    value="A professional, authoritative, and warm male radio presenter introducing a distinguished artist with an elegant tone."
)

def merge_wav_bytes(wav_list):
    """Combines multiple WAV byte streams into a single WAV byte stream."""
    data = []
    params = None
    for w_bytes in wav_list:
        try:
            with wave.open(io.BytesIO(w_bytes), 'rb') as w:
                if params is None:
                    params = w.getparams()
                data.append(w.readframes(w.getnframes()))
        except Exception as e:
            st.error(f"خطا در پردازش فایل صوتی: {e}")
            return None

    output = io.BytesIO()
    with wave.open(output, 'wb') as w:
        w.setparams(params)
        for d in data:
            w.writeframes(d)
    return output.getvalue()

if st.button("🚀 ساخت و ترکیب تیزر کامل", type="primary"):
    if not api_key:
        st.error("لطفاً ابتدا کلید API خود را در نوار کناری وارد کنید.")
    elif not script_text.strip():
        st.error("لطفاً متن معرفی مجری را وارد کنید.")
    else:
        try:
            with st.spinner("در حال تولید صدای مجری توسط Gemini..."):
                client = genai.Client(api_key=api_key)
                
                # Request speech synthesis using tts model or speech config
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"{system_instruction}\n\nRead the following text aloud with appropriate tone and emotion:\n{script_text}",
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=selected_voice
                                )
                            )
                        )
                    )
                )
                
                # Extract host audio
                host_audio_bytes = None
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        host_audio_bytes = part.inline_data.data
                        break
                
                if host_audio_bytes:
                    st.success("✨ صدای معرفی مجری با موفقیت تولید شد!")
                    
                    if uploaded_declination is not None:
                        with st.spinner("در حال ترکیب صدای مجری با دکلمه شما..."):
                            user_audio_bytes = uploaded_declination.read()
                            
                            # Combine host audio + uploaded declination
                            combined_audio = merge_wav_bytes([host_audio_bytes, user_audio_bytes])
                            
                            if combined_audio:
                                st.subheader("🎧 فایل نهایی تیزر (معرفی + دکلمه):")
                                st.audio(combined_audio, format="audio/wav")
                                st.download_button(
                                    label="⬇️ دانلود تیزر کامل (WAV)",
                                    data=combined_audio,
                                    file_name="teaser_ali_karamad.wav",
                                    mime="audio/wav"
                                )
                    else:
                        st.subheader("🎧 صدای معرفی مجری:")
                        st.audio(host_audio_bytes, format="audio/wav")
                        st.info("نکته: برای اینکه دکلمه شما هم به انتهای این صدا متصل شود، فایل دکلمه را در بخش ۲ آپلود کنید.")
                else:
                    st.error("خطا در دریافت خروجی صوتی از Gemini.")
                    
        except Exception as e:
            st.error(f"خطایی رخ داد: {str(e)}")
            
