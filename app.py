import streamlit as st
from google import genai
from google.genai import types
import wave
import io

# تنظیمات اولیه صفحه
st.set_page_config(page_title="استودیوی گویندگی فاخر", page_icon="🎙️", layout="centered")

st.title("🎙️ استودیوی گویندگی و ساخت تیزر رادیویی")
st.caption("تولید صدای مجری با هوش مصنوعی Gemini و ترکیب یکپارچه با دکلمه اختصاصی")

# نوار کناری تنظیمات
with st.sidebar:
    st.header("⚙️ تنظیمات API و گویندگان")
    api_key = st.text_input("کلید API گوگل (Gemini API Key):", type="password")
    
    selected_voice = st.selectbox(
        "صدای مجری (Voice):",
        ["Fenrir", "Puck", "Kore", "Aoede", "Charon"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("💡 کلید API را از [Google AI Studio](https://aistudio.google.com/) دریافت کنید.")

# بخش اول: متن معرفی
st.subheader("۱. متن معرفی مجری")
default_script = "اکنون از پدیده نوظهور در حوزه گویندگی و دکلمه اشعار، با ویژگی اجرای فاخر، جناب آقای علی کارآمد دعوت می‌کنیم برای اجرای یک دکلمه فاخر به جایگاه تشریف بیاورند."
script_text = st.text_area("متن دیالوگ مجری:", value=default_script, height=120)

# بخش دوم: آپلود فایل صوتی
st.subheader("۲. آپلود فایل صوتی دکلمه (استاد علی کارآمد)")
uploaded_declination = st.file_uploader(
    "فایل صوتی دکلمه خود را انتخاب کنید (فرمت WAV):", 
    type=["wav"]
)

def combine_wav_files(host_bytes, user_bytes):
    """
    ترکیب و اتصال فایل صوتی مجری و دکلمه آپلود شده
    """
    try:
        # خواندن فایل اول (مجری)
        host_wav = wave.open(io.BytesIO(host_bytes), 'rb')
        host_params = host_wav.getparams()
        host_frames = host_wav.readframes(host_wav.getnframes())
        host_wav.close()

        # خواندن فایل دوم (دکلمه)
        user_wav = wave.open(io.BytesIO(user_bytes), 'rb')
        user_frames = user_wav.readframes(user_wav.getnframes())
        user_wav.close()

        # یک ثانیه سکوت بین دو صدا
        silence_duration = 1.0 
        silence_frames = b'\x00' * int(host_params.framerate * host_params.nchannels * host_params.sampwidth * silence_duration)

        # ترکیب فریم‌ها
        combined_frames = host_frames + silence_frames + user_frames

        # خروجی فایل WAV نهایی
        output_buffer = io.BytesIO()
        out_wav = wave.open(output_buffer, 'wb')
        out_wav.setparams(host_params)
        out_wav.writeframes(combined_frames)
        out_wav.close()

        return output_buffer.getvalue()
    except Exception as e:
        st.error(f"خطا در ترکیب صوتی: {str(e)}")
        return None

# دکمه اجرای پردازش
if st.button("🚀 ساخت و ترکیب تیزر کامل", type="primary"):
    if not api_key:
        st.error("لطفاً ابتدا کلید API خود را در نوار کناری وارد کنید.")
    elif not script_text.strip():
        st.error("لطفاً متن معرفی مجری را وارد کنید.")
    else:
        try:
            with st.spinner("در حال اتصال به هوش مصنوعی و تولید صدای مجری..."):
                client = genai.Client(api_key=api_key)
                
                # فراخوانی مدل با درخواست صوتی مستقیم
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Please speak the following Persian text clearly as a professional radio presenter:\n\n{script_text.strip()}",
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

                host_audio_bytes = None
                if response and response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.inline_data and part.inline_data.data:
                                    host_audio_bytes = part.inline_data.data
                                    break

                if host_audio_bytes:
                    st.success("✨ صدای معرفی مجری با موفقیت تولید شد!")
                    
                    if uploaded_declination is not None:
                        with st.spinner("در حال الصاق صدای مجری به دکلمه استاد کارآمد..."):
                            user_audio_bytes = uploaded_declination.read()
                            final_audio = combine_wav_files(host_audio_bytes, user_audio_bytes)
                            
                            if final_audio:
                                st.subheader("🎧 تیزر کامل ترکیبی (معرفی مجری + دکلمه):")
                                st.audio(final_audio, format="audio/wav")
                                st.download_button(
                                    label="⬇️ دانلود فایل نهایی تیزر (WAV)",
                                    data=final_audio,
                                    file_name="teaser_ali_karamad_final.wav",
                                    mime="audio/wav"
                                )
                    else:
                        st.subheader("🎧 صدای معرفی مجری:")
                        st.audio(host_audio_bytes, format="audio/wav")
                        st.info("نکته: جهت الصاق دکلمه به انتهای این صدا، فایل WAV دکلمه را در بخش ۲ آپلود کنید.")
                else:
                    st.error("خطا: پاسخی حاوی داده صوتی دریافت نشد. لطفاً کلید API را بررسی کنید.")

        except Exception as e:
            st.error(f"خطای سیستم: {str(e)}")
            
