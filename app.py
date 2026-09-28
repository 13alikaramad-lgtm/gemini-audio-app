import streamlit as st
from google import genai
from google.genai import types
from pydub import AudioSegment
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
        index=0,
        help="صدای Fenrir باوقار و بم است؛ صدای Puck انرژی بیشتری دارد."
    )
    
    st.markdown("---")
    st.markdown("💡 **راهنما:** کلید API خود را از [Google AI Studio](https://aistudio.google.com/) دریافت کنید.")

# بخش اول: متن معرفی
st.subheader("۱. متن معرفی مجری")
default_script = "اکنون از پدیده نوظهور در حوزه گویندگی و دکلمه اشعار، با ویژگی اجرای فاخر، جناب آقای علی کارآمد دعوت می‌کنیم برای اجرای یک دکلمه فاخر به جایگاه تشریف بیاورند."
script_text = st.text_area("متن دیالوگ مجری:", value=default_script, height=120)

# بخش دوم: آپلود فایل صوتی
st.subheader("۲. آپلود فایل صوتی دکلمه (استاد علی کارآمد)")
uploaded_declination = st.file_uploader(
    "فایل صوتی دکلمه خود را انتخاب کنید (فرمت‌های MP3، WAV و M4A پشتیبانی می‌شوند):", 
    type=["wav", "mp3", "m4a", "ogg"]
)

system_instruction = st.text_input(
    "توصیف لحن و شخصیت مجری:",
    value="A professional, authoritative, and warm male radio presenter introducing a distinguished artist with an elegant tone."
)

def combine_audio_files(genai_audio_bytes, user_file_bytes):
    """
    ترکیب و همگام‌سازی فرکانس دو فایل صوتی با استفاده از Pydub
    """
    try:
        # بارگذاری صدای مجری از Gemini (معمولاً فرمت WAV است)
        host_segment = AudioSegment.from_file(io.BytesIO(genai_audio_bytes))
        
        # بارگذاری صدای آپلودشده کاربر
        user_segment = AudioSegment.from_file(io.BytesIO(user_file_bytes))
        
        # یکسان‌سازی استاندارد صوتی (نرخ نمونه‌برداری 44.1kHz، کانال استریو)
        host_segment = host_segment.set_frame_rate(44100).set_channels(2)
        user_segment = user_segment.set_frame_rate(44100).set_channels(2)
        
        # ایجاد ۱ ثانیه سکوت بین معرفی مجری و آغاز دکلمه جهت زیبایی اجرا
        silence = AudioSegment.silent(duration=1000)
        
        # ترکیب صوتی
        combined = host_segment + silence + user_segment
        
        # خروجی به صورت بایتی
        output_buffer = io.BytesIO()
        combined.export(output_buffer, format="wav")
        return output_buffer.getvalue()
        
    except Exception as e:
        st.error(f"خطا در همگام‌سازی و ترکیب فایل صوتی: {str(e)}")
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
                
                # فراخوانی استاندارد مدل تولید صوت Gemini
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
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
                
                # استخراج داده صوتی تولیدشده
                host_audio_bytes = None
                if response.candidates and len(response.candidates) > 0:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            host_audio_bytes = part.inline_data.data
                            break
                
                if host_audio_bytes:
                    st.success("✨ صدای معرفی مجری با موفقیت تولید شد!")
                    
                    if uploaded_declination is not None:
                        with st.spinner("در حال یکسان‌سازی فرکانس و ترکیب هوشمند صدای مجری با دکلمه شما..."):
                            user_audio_bytes = uploaded_declination.read()
                            
                            # ترکیب استاندارد صوتی
                            final_audio = combine_audio_files(host_audio_bytes, user_audio_bytes)
                            
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
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "400" in err_msg:
                st.error("کلید API گوگل نامعتبر است یا دسترسی آن فعال نیست. لطفاً کلید جدیدی دریافت کنید.")
            else:
                st.error(f"خطای سیستم: {err_msg}")
