import streamlit as st
import yt_dlp
import librosa
import numpy as np
import os
import re

# --- הגדרות עיצוב (CSS) ---
st.markdown("""
    <style>
    .stApp { direction: rtl; text-align: right; }
    h1, h2, h3, p, div, label, span, .stMarkdown { text-align: right; }
    .stTextInput > div > div > input { direction: ltr; text-align: left; } 
    textarea { direction: rtl; text-align: right; font-family: 'Courier New', monospace; }
    .stButton > button { width: 100%; }
    
    /* כרטיסיות אקורדים */
    .chord-card {
        display: inline-block;
        margin: 5px;
        padding: 10px;
        color: white;
        border-radius: 8px;
        text-align: center;
        min-width: 60px;
    }
    
    /* תיבת קאפו */
    .capo-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ffeeba;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# חלק א': לוגיקה לניתוח שירים (DSP & AI)
# ==========================================

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TEMPLATES = {}

for i, note in enumerate(NOTES):
    # Major
    vec = np.zeros(12); vec[i]=1; vec[(i+4)%12]=1; vec[(i+7)%12]=1
    CHORD_TEMPLATES[note] = vec
    # Minor
    vec = np.zeros(12); vec[i]=1; vec[(i+3)%12]=1; vec[(i+7)%12]=1
    CHORD_TEMPLATES[note + 'm'] = vec

def identify_chord(chroma_vector):
    best_chord = None
    max_score = -1
    for chord_name, template in CHORD_TEMPLATES.items():
        score = np.dot(chroma_vector, template)
        if score > max_score:
            max_score = score
            best_chord = chord_name
    return best_chord

def download_audio(youtube_url):
    # בדיקה חכמה: אם יש קובץ EXE (במחשב שלך) תשתמש בו, אחרת (בענן) תן למערכת למצוא לבד
    ffmpeg_local = os.path.join(os.getcwd(), 'ffmpeg.exe')
    ffmpeg_location = os.getcwd() if os.path.exists(ffmpeg_local) else None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}],
        'quiet': True,
        'ffmpeg_location': ffmpeg_location # השינוי כאן
    }
    # ... המשך הפונקציה אותו דבר ...
    if os.path.exists("temp_audio.mp3"): os.remove("temp_audio.mp3")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return "temp_audio.mp3"

def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, duration=90) 
    y_harmonic, _ = librosa.effects.hpss(y)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    chords = []
    fps = sr / 512
    step = int(fps) 
    
    for i in range(0, chromagram.shape[1], step):
        chord = identify_chord(chromagram[:, i])
        timestamp = int(i / fps)
        if not chords or chords[-1][1] != chord:
            chords.append((timestamp, chord))
    return chords

# ==========================================
# חלק ב': לוגיקה לטרנספוזיציה (Text Processing)
# ==========================================

NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_TO_SHARP = {'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#', 'Cb':'B', 'Fb':'E'}

def transpose_text_logic(text, semitones):
    pattern = r"\b([A-G](?:#|b)?)(m|maj|min|dim|aug|sus|add|7|9|11|13|5)*\b"
    
    def replace(match):
        full_chord = match.group(0)
        base = match.group(1)
        suffix = match.group(2) if match.group(2) else ""
        
        base_sharp = FLAT_TO_SHARP.get(base, base)
        if base_sharp in NOTES_SHARP:
            idx = NOTES_SHARP.index(base_sharp)
            new_base = NOTES_SHARP[(idx + semitones) % 12]
            return new_base + suffix
        return full_chord

    return re.sub(pattern, replace, text)

# ==========================================
# הממשק הראשי (Main App)
# ==========================================

# --- סרגל צד: ניווט ומילון ---
with st.sidebar:
    st.title("🎸 כלי הנגינה שלי")
    app_mode = st.radio("בחר כלי:", ["ניתוח שיר מיוטיוב", "עורך שירים (Transpose)"])
    
    st.markdown("---")
    st.header("📖 מילון אקורדים")
    r_col, t_col = st.columns(2)
    root = r_col.selectbox("תו", NOTES_SHARP)
    type_ = t_col.selectbox("סוג", ["Major", "m", "7", "m7", "maj7"])
    
    # תצוגת התמונה
    clean_root = root.replace("#", "%23")
    st.image(f"https://chord-api.v0.app/api/chords/{type_}/{clean_root}", caption=f"פוזיציה ל-{root}{type_}")

# --- מסך א': מנתח יוטיוב ---
if app_mode == "ניתוח שיר מיוטיוב":
    st.title('🎧 מנתח שירים מיוטיוב')
    st.write("מזהה אקורדים (כולל Minor) ישירות מהאזנה לשיר.")
    
    url = st.text_input('הדבק לינק לשיר:')
    if url and st.button('נתח שיר'):
        try:
            with st.spinner('מוריד ומעבד (זה לוקח רגע)...'):
                audio = download_audio(url)
                chords = analyze_audio(audio)
            
            st.success('הניתוח הסתיים!')
            st.audio(audio)
            
            st.subheader("🎼 ציר זמן של השיר:")
            html = ""
            for time, chord in chords:
                color = "#2196F3" if 'm' in chord else "#4CAF50" 
                html += f"""
                <div class="chord-card" style="background-color: {color};">
                    <div style="font-size: 12px;">{time}s</div>
                    <div style="font-size: 20px; font-weight: bold;">{chord}</div>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"שגיאה: {e}")
            st.info("טיפ: וודא שקבצי FFmpeg נמצאים בתיקייה.")

# --- מסך ב': עורך שירים ---
elif app_mode == "עורך שירים (Transpose)":
    st.title('📝 עורך ומשנה סולמות')
    st.write("הדבק טקסט עם אקורדים ושנה את הטון בקליק.")
    
    if 'transpose' not in st.session_state: st.session_state.transpose = 0
    
    text_in = st.text_area("הדבק כאן שיר:", height=150, placeholder="Am      G\nשלום לך...")
    
    c1, c2, c3 = st.columns([1,2,1])
    if c1.button("➖ הורד חצי טון"): st.session_state.transpose -= 1
    if c3.button("➕ העלה חצי טון"): st.session_state.transpose += 1
    if c2.button("איפוס"): st.session_state.transpose = 0
    
    # === הלוגיקה החדשה והמשופרת של הקאפו ===
    shift = st.session_state.transpose
    capo_msg = ""
    
    if shift == 0:
        capo_msg = "אתה בטון המקורי (ללא קאפו)."
    else:
        # חישוב מתמטי: אם עלינו ב-X, זה כמו לרדת ב-(12 פחות X)
        # דוגמה: עלינו ב-2 (טון). כדי לחזור למקור צריך "להשלים" ל-12. אז 10.
        # דוגמה: ירדנו ב-2 (מינוס 2). הערך המוחלט הוא 2.
        
        if shift < 0:
            capo_fret = abs(shift)
        else:
            capo_fret = 12 - (shift % 12)
            if capo_fret == 12: capo_fret = 0 # מקרה קצה
            
        capo_msg = f"💡 כדי לנגן עם האקורדים האלו בטון המקורי: **קאפו בשריג {capo_fret}**"

    st.markdown(f"""
    <div class="capo-box">
        Shift: {shift}<br>
        {capo_msg}
    </div>
    """, unsafe_allow_html=True)
    # ==============================
    
    if text_in:
        st.markdown("---")
        new_text = transpose_text_logic(text_in, st.session_state.transpose)
        # הוספתי כאן color: #1e1e1e כדי להכריח צבע טקסט כהה
        st.markdown(f"""
        <div style='background:#f0f2f6; color:#1e1e1e; padding:20px; border-radius:10px; white-space:pre-wrap; font-family:monospace; direction:rtl; font-size: 16px;'>{new_text}</div>
        """, unsafe_allow_html=True)