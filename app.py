import streamlit as st
from google import genai
from google.genai import types
import wave
import io

# تنظیمات صفحه
st.set_page_config(page_title="استودیوی گویندگی فاخر", page_icon="🎙️", layout="centered")

st.title("🎙️ استودیوی گویندگی و ساخت تیزر رادیویی")
st.caption("تولید صدای مجری با هوش مصنوعی Gemini و ترکیب یکپارچه با دکلمه اختصاصی")

# نوار تنظیمات
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

system_instruction = st.text_input(
    "توصیف لحن و شخصیت مجری:",
    value="A professional, authoritative, and warm male radio presenter introducing a distinguished artist with an elegant tone."
)

def combine_wav_streams(host_bytes, user_bytes):
    try:
        host_wav = wave.open(io.BytesIO(host_bytes), 'rb')
        user_wav = wave.open(io.BytesIO(user_bytes), 'rb')
        
        params = host_wav.getparams()
        
        host_frames = host_wav.readframes(host_wav.getnframes())
        user_frames = user_wav.readframes(user_wav.getnframes())
        
        # سکوت ۱ ثانیه‌ای بین صداها
        silence_frames = b'\x00' * (params.framerate * params.nchannels * params.sampwidth)
        
        combined_frames = host_frames + silence_frames + user_frames
        
        output = io.BytesIO()
        out_wav = wave.open(output, 'wb')
        out_wav.setparams(params)
        out_wav.writeframes(combined_frames)
        out_wav.close()
        
        return output.getvalue()
    except Exception as e:
        st.error(f"خطا در ترکیب فایل صوتی: {str(e)}")
        return None

# اجرای پردازش
if st.button("🚀 ساخت و ترکیب تیزر کامل", type="primary"):
    if not api_key:
        st.error("لطفاً ابتدا کلید API خود را در نوار کناری وارد کنید.")
    elif not script_text.strip():
        st.error("لطفاً متن معرفی مجری را وارد کنید.")
    else:
        try:
            with st.spinner("در حال اتصال به هوش مصنوعی و تولید صدای مجری..."):
                client = genai.Client(api_key=api_key)
                
                # استفاده از مدل پایدار و فعال gemini-2.5-flash
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"{system_instruction}\n\nRead the following text aloud with high elegance:\n{script_text}",
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
                if response.candidates and len(response.candidates) > 0:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            host_audio_bytes = part.inline_data.data
                            break
                
                if host_audio_bytes:
                    st.success("✨ صدای معرفی مجری با موفقیت تولید شد!")
                    
                    if uploaded_declination is not None:
                        with st.spinner("در حال ترکیب صدای مجری با دکلمه شما..."):
                            user_audio_bytes = uploaded_declination.read()
                            final_audio = combine_wav_streams(host_audio_bytes, user_audio_bytes)
                            
                            if final_audio:
                                st.subheader("🎧 تیزر کامل ترکیبی (معرفی مجری + دکلمه استاد کارآمد):")
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
                        st.info("نکته: برای اینکه دکلمه شما هم به انتهای این صدا متصل شود، فایل دکلمه را در بخش ۲ آپلود کنید.")
                else:
                    st.error("خطا: پاسخی حاوی داده صوتی از گوگل دریافت نشد.")
                    
        except Exception as e:
            st.error(f"خطای سیستم: {str(e)}")
            
