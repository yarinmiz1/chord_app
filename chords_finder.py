import streamlit as st
import streamlit.components.v1 as components
import yt_dlp
import librosa
import numpy as np
import os
import re

# ==========================================
# ניהול מצבים (Session State) - אתחול
# ==========================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'transpose' not in st.session_state:
    st.session_state.transpose = 0
if 'simplify' not in st.session_state:
    st.session_state.simplify = False
if 'is_easy_mode' not in st.session_state:
    st.session_state.is_easy_mode = False
if 'scroll_speed' not in st.session_state:
    st.session_state.scroll_speed = 0

# ==========================================
# עיצוב דינמי (Light / Dark Mode)
# ==========================================
st.sidebar.title("⚙️ הגדרות")
st.session_state.dark_mode = st.sidebar.toggle("🌙 מצב כהה / ☀️ מצב בהיר", value=st.session_state.dark_mode)

bg_color = "#1e1e1e" if st.session_state.dark_mode else "#ffffff"
text_color = "#ffffff" if st.session_state.dark_mode else "#1e1e1e"
box_bg = "#2b2b2b" if st.session_state.dark_mode else "#f0f2f6"
border_color = "#d9534f" if st.session_state.dark_mode else "#4a90e2" 
tooltip_bg = "#ffffff" if st.session_state.dark_mode else "#333333"
tooltip_text = "#000000" if st.session_state.dark_mode else "#ffffff"

btn_bg = "#d9534f" if st.session_state.dark_mode else "#4a90e2"
btn_hover_shadow = "rgba(217, 83, 79, 0.4)" if st.session_state.dark_mode else "rgba(74, 144, 226, 0.4)"

st.markdown(f"""
    <style>
    .stApp {{ direction: rtl; text-align: right; background-color: {bg_color}; color: {text_color}; }}
    h1, h2, h3, p, div, label, span, .stMarkdown {{ text-align: right; color: {text_color}; }}
    .stTextInput > div > div > input {{ direction: ltr; text-align: left; }} 
    
    [data-testid="stSidebar"] {{ background-color: {box_bg} !important; }}
    [data-testid="stSidebar"] * {{ color: {text_color} !important; }}
    
    textarea {{ 
        direction: rtl; text-align: right; font-family: 'Courier New', monospace; 
        background-color: {box_bg} !important; color: {text_color} !important;
        border: 1px solid {border_color} !important; border-radius: 10px !important;
    }}
    
    .stButton > button {{ 
        width: 100%; background: {btn_bg}; color: white !important;
        border-radius: 8px; border: none; transition: all 0.3s ease; font-weight: bold;
    }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px {btn_hover_shadow}; }}
    
    .chord-card {{ display: inline-block; margin: 5px; padding: 10px; color: white !important; border-radius: 8px; text-align: center; min-width: 60px; }}
    
    .capo-box {{ background-color: {box_bg}; color: {border_color} !important; padding: 15px; border-radius: 10px; border-left: 5px solid {border_color}; text-align: center; margin-top: 10px; margin-bottom: 10px; font-weight: bold; font-size: 18px; }}
    
    .chord-hover {{ position: relative; display: inline-block; color: {border_color} !important; font-weight: bold; cursor: pointer; border-bottom: 2px solid {border_color}; padding: 0 2px; }}
    
    .chord-tooltip {{ 
        visibility: hidden; background-color: {tooltip_bg}; color: {tooltip_text}; 
        text-align: center; border-radius: 12px; padding: 10px; position: absolute; 
        z-index: 10; bottom: 140%; left: 50%;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4); opacity: 0; transition: opacity 0.2s; 
    }}
    .chord-tooltip img, .chord-tooltip svg {{ width: 100%; display: block; border-radius: 5px; }}
    .chord-tooltip::after {{ content: ""; position: absolute; top: 100%; left: 50%; margin-left: -8px; border-width: 8px; border-style: solid; border-color: {tooltip_bg} transparent transparent transparent; }}
    .chord-hover:hover .chord-tooltip {{ visibility: visible; opacity: 1; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# לוגיקה מוזיקלית
# ==========================================
NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_TO_SHARP = {'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#', 'Cb':'B', 'Fb':'E'}

CHORD_FRETS = {
    'C': 'x32010', 'C#': 'x46664', 'D': 'xx0232', 'D#': 'x68886', 'E': '022100', 'F': '133211', 'F#': '244322', 
    'G': '320003', 'G#': '466544', 'A': 'x02220', 'A#': 'x13331', 'B': 'x24442',
    'Cm': 'x35543', 'C#m': 'x46654', 'Dm': 'xx0231', 'D#m': 'x68876', 'Em': '022000', 'Fm': '133111', 'F#m': '244222', 
    'Gm': '355333', 'G#m': '466444', 'Am': 'x02210', 'A#m': 'x13321', 'Bm': 'x24432'
}

def get_index(n):
    n = FLAT_TO_SHARP.get(n, n)
    return NOTES_SHARP.index(n) if n in NOTES_SHARP else -1

def shift_base(base, semitones):
    idx = get_index(base)
    if idx == -1: return base
    return NOTES_SHARP[(idx + semitones) % 12]

def simplify_suffix(suffix):
    if not suffix: return ""
    if suffix.startswith('m') and not suffix.startswith('maj'): return "m"
    return ""

def generate_piano_svg(base, suffix):
    if base not in NOTES_SHARP: return ""
    root_idx = NOTES_SHARP.index(base)
    
    if 'm' in suffix and 'maj' not in suffix: intervals = [0, 3, 7]
    else: intervals = [0, 4, 7]
        
    active_keys = [(root_idx + i) for i in intervals]
    
    white_key_indices = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]
    black_key_indices = [1, 3, 6, 8, 10, 13, 15, 18, 20, 22]
    black_offsets = {1: 1, 3: 2, 6: 4, 8: 5, 10: 6, 13: 8, 15: 9, 18: 11, 20: 12, 22: 13}
    
    svg = '<svg viewBox="0 0 280 60" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="background:#fff; border-radius:4px; padding:2px;">'
    
    white_w, white_h = 20, 60
    for i, w_idx in enumerate(white_key_indices):
        fill = "#ff4b4b" if w_idx in active_keys else "#ffffff"
        svg += f'<rect x="{i*white_w}" y="0" width="{white_w}" height="{white_h}" fill="{fill}" stroke="#333" stroke-width="1"/>'
        
    black_w, black_h = 12, 35
    for b_idx in black_key_indices:
        boundary = black_offsets[b_idx]
        x_pos = boundary * white_w - (black_w / 2)
        fill = "#ff4b4b" if b_idx in active_keys else "#000000"
        svg += f'<rect x="{x_pos}" y="0" width="{black_w}" height="{black_h}" fill="{fill}" stroke="#333" stroke-width="1"/>'
        
    svg += '</svg>'
    return svg

def transpose_text_logic(text, semitones, simplify=False, instrument="🎸 גיטרה"):
    pattern = r"\b([A-G](?:#|b)?)(m|maj|min|dim|aug|sus|add|7|9|11|13|5)*\b"
    def replace(match):
        full_chord, base, suffix = match.group(0), match.group(1), match.group(2) or ""
        new_base = shift_base(base, semitones)
        new_suffix = simplify_suffix(suffix) if simplify else suffix
        new_chord = new_base + new_suffix

        # הגדלת ה-Tooltip דינמית בהתאם לכלי הנבחר
        if "פסנתר" in instrument:
            tooltip_content = generate_piano_svg(new_base, simplify_suffix(suffix))
            t_style = "width: 340px; margin-left: -170px;" # פסנתר רחב וגדול
        else:
            basic_chord = new_base + simplify_suffix(suffix)
            frets = CHORD_FRETS.get(basic_chord, 'xxxxxx')
            clean_root = new_chord.replace("#", "%23")
            img_url = f"https://chordgenerator.net/{clean_root}.png?p={frets}&s=10"
            tooltip_content = f'<img src="{img_url}" alt="{new_chord}">'
            t_style = "width: 220px; margin-left: -110px;" # גיטרה בגודל רגיל
            
        return f'<span class="chord-hover">{new_chord}<span class="chord-tooltip" style="{t_style}">{tooltip_content}</span></span>'
    return re.sub(pattern, replace, text)

def find_easy_shift(text):
    best_shift, max_easy_count = 0, -1
    easy_chords = {'C', 'Am', 'G', 'F', 'Em', 'Dm', 'E', 'A', 'D'} 
    found_chords = re.findall(r"\b([A-G](?:#|b)?)(m|maj|min|dim|aug|sus|add|7|9|11|13|5)*\b", text)

    for shift in range(12):
        easy_count = sum(1 for b, s in found_chords if shift_base(b, shift) + simplify_suffix(s) in easy_chords)
        if easy_count > max_easy_count: max_easy_count, best_shift = easy_count, shift
    return best_shift if best_shift <= 6 else best_shift - 12

# ==========================================
# הממשק הראשי - שימוש במשתנים קבועים כדי למנוע באגים של אי התאמה!
# ==========================================
MENU_EDITOR = "🎶 עורך ומשנה סולמות"
MENU_YOUTUBE = "🎧 מנתח שירים מיוטיוב"

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("בחר כלי:", [MENU_EDITOR, MENU_YOUTUBE])

if app_mode == MENU_EDITOR:
    st.title('🎶 עורך ומשנה סולמות')
    text_in = st.text_area("הדבק כאן שיר (מאוחד עם האקורדים):", height=150, placeholder="Am      G\nשלום לך...")
    
    c1, c2, c3 = st.columns([1,2,1])
    if c1.button("➖ הורד חצי טון"): 
        st.session_state.transpose -= 1
        st.session_state.is_easy_mode = False
    if c3.button("➕ העלה חצי טון"): 
        st.session_state.transpose += 1
        st.session_state.is_easy_mode = False
    if c2.button("איפוס סולם"): 
        st.session_state.transpose = 0
        st.session_state.is_easy_mode = False
    
    c4, c5 = st.columns(2)
    if st.session_state.is_easy_mode:
        if c4.button("🔙 חזרה לגרסה המקורית"):
            st.session_state.transpose = 0
            st.session_state.is_easy_mode = False
            st.rerun()
    else:
        if c4.button("🎸 מצא גרסה קלה לניגון"):
            if text_in:
                st.session_state.transpose = find_easy_shift(text_in)
                st.session_state.is_easy_mode = True
                st.rerun()
            else:
                st.warning("קודם הדבק שיר כדי שאוכל למצוא גרסה קלה!")
            
    btn_simplify_text = "🔙 החזר אקורדים מורכבים" if st.session_state.simplify else "🪄 פישוט אקורדים "
    if c5.button(btn_simplify_text):
        st.session_state.simplify = not st.session_state.simplify
    
    shift = st.session_state.transpose
    capo_msg = "ללא קאפו (טון מקורי)" if shift == 0 else f"💡 לנגינה בטון המקורי: קאפו בשריג {abs(shift) if shift < 0 else 12 - (shift % 12)}"
    if shift > 0 and (12 - (shift % 12)) == 12: capo_msg = "ללא קאפו (טון מקורי)"
    st.markdown(f'<div class="capo-box">תזוזה: {shift} חצאי טונים<br>{capo_msg}</div>', unsafe_allow_html=True)
    
    if text_in:
        st.markdown("---")
        
        instrument = st.radio("תצוגת אקורדים במעבר עכבר:", ["🎸 גיטרה", "🎹 פסנתר"], horizontal=True)
        
        st.markdown("### 🐢 מהירות גלילה אוטומטית:")
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        if sc1.button("➖ האט"): st.session_state.scroll_speed = max(0, st.session_state.scroll_speed - 1)
        if sc3.button("➕ האץ"): st.session_state.scroll_speed += 1
        sc2.markdown(f"<h3 style='text-align:center;'>{st.session_state.scroll_speed}</h3>", unsafe_allow_html=True)
        
        if st.session_state.scroll_speed > 0:
            speed = st.session_state.scroll_speed
            components.html(f"""
                <script>
                    const speed = {speed};
                    setInterval(() => {{
                        const pDoc = window.parent.document;
                        const containers = [
                            pDoc.querySelector('[data-testid="stAppViewContainer"]'),
                            pDoc.querySelector('.main'),
                            pDoc.documentElement
                        ];
                        for (let c of containers) {{
                            if (c && c.scrollHeight > c.clientHeight) {{
                                c.scrollTop += speed;
                                break;
                            }}
                        }}
                    }}, 50);
                </script>
            """, height=0)

        new_text = transpose_text_logic(text_in, st.session_state.transpose, simplify=st.session_state.simplify, instrument=instrument)
        st.markdown(f"<div style='background:{box_bg}; color:{text_color}; padding:25px; border-radius:15px; border: 1px solid {border_color}; white-space:pre-wrap; direction:rtl; font-size: 18px; line-height: 1.8;'>{new_text}</div>", unsafe_allow_html=True)

elif app_mode == MENU_YOUTUBE:
    st.title('🎧 מנתח שירים מיוטיוב')
    url = st.text_input('הדבק לינק לשיר:')
    if url and st.button('נתח שיר'):
        try:
            with st.spinner('מוריד ומעבד (זה לוקח רגע)...'):
                def identify_chord_local(chroma_vector):
                    best, m_score = None, -1
                    for i, note in enumerate(NOTES_SHARP):
                        vec = np.zeros(12); vec[i]=1; vec[(i+4)%12]=1; vec[(i+7)%12]=1
                        score = np.dot(chroma_vector, vec)
                        if score > m_score: m_score, best = score, note
                        vec = np.zeros(12); vec[i]=1; vec[(i+3)%12]=1; vec[(i+7)%12]=1
                        score = np.dot(chroma_vector, vec)
                        if score > m_score: m_score, best = score, note + 'm'
                    return best

                ydl_opts = {'format': 'bestaudio/best', 'outtmpl': 'temp_audio.%(ext)s', 'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}], 'quiet': True}
                if os.path.exists("temp_audio.mp3"): os.remove("temp_audio.mp3")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
                
                y, sr = librosa.load("temp_audio.mp3", duration=90) 
                y_harmonic, _ = librosa.effects.hpss(y)
                chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
                
                chords, fps = [], sr / 512
                for i in range(0, chromagram.shape[1], int(fps)):
                    chord = identify_chord_local(chromagram[:, i])
                    time = int(i / fps)
                    if not chords or chords[-1][1] != chord: chords.append((time, chord))

            st.success('הניתוח הסתיים!')
            st.audio("temp_audio.mp3")
            
            html = ""
            for time, chord in chords:
                color = border_color if 'm' in chord else "#4CAF50" 
                html += f'<div class="chord-card" style="background-color: {color}; color: white !important;"><div style="font-size: 12px;">{time}s</div><div style="font-size: 20px; font-weight: bold;">{chord}</div></div>'
            st.markdown(html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"שגיאה: {e}")
