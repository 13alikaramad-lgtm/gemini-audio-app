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

prompt_instruction = st.text_input(
    "توصیف لحن و شخصیت مجری:",
    value="A professional, authoritative, and warm male radio presenter introducing a distinguished artist with an elegant tone."
)

def ensure_wav_header(raw_bytes, sample_rate=24000, num_channels=1, sample_width=2):
    """
    بررسی و افزودن هدر استاندارد WAV به داده‌های صوتی خام در صورت لزوم
    """
    try:
        # اگر فایل از قبل هدر RIFF/WAV معتبر دارد
        with wave.open(io.BytesIO(raw_bytes), 'rb') as w:
            return raw_bytes, w.getparams()
    except Exception:
        # ساخت هدر WAV برای داده صوتی raw PCM
        out_stream = io.BytesIO()
        with wave.open(out_stream, 'wb') as w:
            w.setnchannels(num_channels)
            w.setsampwidth(sample_width)
            w.setframerate(sample_rate)
            w.writeframes(raw_bytes)
        out_stream.seek(0)
        
        with wave.open(out_stream, 'rb') as w:
            params = w.getparams()
            
        return out_stream.getvalue(), params

def combine_wav_bytes(host_raw_bytes, user_raw_bytes):
    """
    ترکیب و الصاق دقیق دو فایل صوتی همراه با ۱ ثانیه سکوت بین آن‌ها
    """
    try:
        host_wav_bytes, host_params = ensure_wav_header(host_raw_bytes)
        
        host_w = wave.open(io.BytesIO(host_wav_bytes), 'rb')
        user_w = wave.open(io.BytesIO(user_raw_bytes), 'rb')
        
        host_frames = host_w.readframes(host_w.getnframes())
        user_frames = user_w.readframes(user_w.getnframes())
        
        # ساخت ۱ ثانیه سکوت بر اساس پارامترهای صدای مجری
        silence_duration_sec = 1.0
        silence_frames_count = int(host_params.framerate * silence_duration_sec)
        silence_bytes = b'\x00' * (silence_frames_count * host_params.nchannels * host_params.sampwidth)
        
        combined_frames = host_frames + silence_bytes + user_frames
        
        out_buffer = io.BytesIO()
        with wave.open(out_buffer, 'wb') as out_w:
            out_w.setparams(host_params)
            out_w.writeframes(combined_frames)
            
        return out_buffer.getvalue()
    except Exception as e:
        st.error(f"خطا در الصاق فایل‌های صوتی: {str(e)}")
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
                
                # تنظیم دستورالعمل لحن در system_instruction (جدا از متن اصلی)
                config = types.GenerateContentConfig(
                    system_instruction=prompt_instruction,
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=selected_voice
                            )
                        )
                    )
                )
                
                # لیست مدل‌های صوتی پشتیبانی شده برای فراخوانی
                models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']
                response = None
                
                for model_name in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=script_text.strip(), # فقط متن فارسی برای خواندن ارسال می‌شود
                            config=config
                        )
                        if response:
                            break
                    except Exception:
                        continue

                host_audio_bytes = None
                if response and response.candidates and len(response.candidates) > 0:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            host_audio_bytes = part.inline_data.data
                            break
                
                if host_audio_bytes:
                    st.success("✨ صدای معرفی مجری با موفقیت تولید شد!")
                    
                    if uploaded_declination is not None:
                        with st.spinner("در حال الصاق صدای مجری به دکلمه آپلود شده..."):
                            user_audio_bytes = uploaded_declination.read()
                            final_audio = combine_wav_bytes(host_audio_bytes, user_audio_bytes)
                            
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
                        st.info("نکته: برای اینکه دکلمه شما هم به انتهای این صدا متصل شود، فایل WAV دکلمه را در بخش ۲ آپلود کنید.")
                else:
                    st.error("خطا: پاسخی حاوی داده صوتی از گوگل دریافت نشد.")
                    
        except Exception as e:
            st.error(f"خطای سیستم: {str(e)}")
