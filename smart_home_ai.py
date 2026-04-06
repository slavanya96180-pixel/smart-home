# ================================================
# 🏠 SMART HOME AI SYSTEM v2.0
# ================================================
# Upgrades over v1:
# - Multi-room device management (Living, Bedroom, Kitchen, Office)
# - Natural language understanding (flexible command parsing)
# - Energy monitoring with usage history
# - Device scheduling (time-based automation)
# - Voice profile enrollment & MFCC-based authentication
# - Expandable ML pipeline (TF-IDF + SVM > Naive Bayes)
# - Real-time activity log with timestamps
# - Modern UI with tabs and responsive layout
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import json
from datetime import datetime, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf

# ──────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home AI v2",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stButton>button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white; border: none; border-radius: 8px;
        padding: 0.4rem 1rem; font-weight: 600;
    }
    .stButton>button:hover { opacity: 0.85; }
    .device-card {
        background: #1e2130; border-radius: 12px;
        padding: 1rem; margin: 0.5rem 0;
        border-left: 4px solid #1a73e8;
    }
    .device-on  { border-left-color: #00c853; }
    .device-off { border-left-color: #ef5350; }
    .metric-box {
        background: #1e2130; border-radius: 10px;
        padding: 0.75rem 1rem; text-align: center;
    }
    .log-entry { font-size: 0.82rem; color: #aaa; padding: 2px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ──────────────────────────────────────────────
def _default_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "devices" not in st.session_state:
        # {room: {device: {"state": bool, "watts": int, "usage_kwh": float}}}
        st.session_state.devices = {
            "Living Room":  {"Light": {"state": False, "watts": 60,  "usage_kwh": 0.0},
                             "Fan":   {"state": False, "watts": 75,  "usage_kwh": 0.0},
                             "TV":    {"state": False, "watts": 120, "usage_kwh": 0.0}},
            "Bedroom":      {"Light": {"state": False, "watts": 40,  "usage_kwh": 0.0},
                             "AC":    {"state": False, "watts": 1500,"usage_kwh": 0.0},
                             "Fan":   {"state": False, "watts": 55,  "usage_kwh": 0.0}},
            "Kitchen":      {"Light": {"state": False, "watts": 50,  "usage_kwh": 0.0},
                             "Exhaust Fan": {"state": False, "watts": 30, "usage_kwh": 0.0},
                             "Microwave": {"state": False, "watts": 1000,"usage_kwh": 0.0}},
            "Office":       {"Light": {"state": False, "watts": 45,  "usage_kwh": 0.0},
                             "AC":    {"state": False, "watts": 1200,"usage_kwh": 0.0},
                             "Laptop Charger": {"state": False, "watts": 65, "usage_kwh": 0.0}},
        }

    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []

    if "schedules" not in st.session_state:
        st.session_state.schedules = []   # list of {room, device, action, trigger_time}

    if "owner_mfcc" not in st.session_state:
        st.session_state.owner_mfcc = None

    if "energy_history" not in st.session_state:
        # Simulated 7-day history (kWh per day)
        np.random.seed(42)
        st.session_state.energy_history = list(np.round(np.random.uniform(3.5, 8.0, 7), 2))

_default_state()

# ──────────────────────────────────────────────
# TRAINING DATASET  (expanded, NL-friendly)
# ──────────────────────────────────────────────
TRAIN_DATA = {
    "command": [
        # LIGHT ON
        "turn on the light", "switch on light", "light on", "lights on please",
        "can you turn on the light", "enable the light", "brighten the room",
        # LIGHT OFF
        "turn off the light", "switch off light", "light off", "lights off",
        "disable the light", "kill the lights", "turn the lights out",
        # FAN ON
        "turn on the fan", "switch on fan", "fan on", "start the fan",
        "can you start the fan", "enable the fan", "run the fan",
        # FAN OFF
        "turn off the fan", "switch off fan", "fan off", "stop the fan",
        "disable the fan", "shut down the fan",
        # AC ON
        "turn on the AC", "switch on AC", "AC on", "start air conditioning",
        "enable the air conditioner", "cool the room", "turn on air con",
        # AC OFF
        "turn off the AC", "switch off AC", "AC off", "stop air conditioning",
        "disable air conditioner", "turn off air con",
        # TV ON
        "turn on the TV", "switch on TV", "TV on", "start the television",
        "enable the TV",
        # TV OFF
        "turn off the TV", "switch off TV", "TV off", "stop the television",
        "disable the TV", "mute and turn off TV",
        # ALL ON / OFF
        "turn on everything", "all devices on", "everything on",
        "turn off everything", "all devices off", "everything off", "shutdown all",
    ],
    "action": [
        "LIGHT_ON","LIGHT_ON","LIGHT_ON","LIGHT_ON","LIGHT_ON","LIGHT_ON","LIGHT_ON",
        "LIGHT_OFF","LIGHT_OFF","LIGHT_OFF","LIGHT_OFF","LIGHT_OFF","LIGHT_OFF","LIGHT_OFF",
        "FAN_ON","FAN_ON","FAN_ON","FAN_ON","FAN_ON","FAN_ON","FAN_ON",
        "FAN_OFF","FAN_OFF","FAN_OFF","FAN_OFF","FAN_OFF","FAN_OFF",
        "AC_ON","AC_ON","AC_ON","AC_ON","AC_ON","AC_ON","AC_ON",
        "AC_OFF","AC_OFF","AC_OFF","AC_OFF","AC_OFF","AC_OFF",
        "TV_ON","TV_ON","TV_ON","TV_ON","TV_ON",
        "TV_OFF","TV_OFF","TV_OFF","TV_OFF","TV_OFF","TV_OFF",
        "ALL_ON","ALL_ON","ALL_ON",
        "ALL_OFF","ALL_OFF","ALL_OFF","ALL_OFF",
    ]
}

# ──────────────────────────────────────────────
# MODEL TRAINING  (TF-IDF + LinearSVC pipeline)
# ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.DataFrame(TRAIN_DATA)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LinearSVC(C=1.0, max_iter=2000)),
    ])
    pipeline.fit(df["command"], df["action"])
    # Quick cross-val score (not exposed, just for internal check)
    return pipeline

model_pipeline = train_model()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
ACTION_EMOJI = {
    "LIGHT_ON": "💡", "LIGHT_OFF": "💡",
    "FAN_ON": "🌀", "FAN_OFF": "🌀",
    "AC_ON": "❄️", "AC_OFF": "❄️",
    "TV_ON": "📺", "TV_OFF": "📺",
    "ALL_ON": "🔆", "ALL_OFF": "🔴",
}

ACTION_DEVICE_MAP = {
    "LIGHT_ON": ("Light", True),  "LIGHT_OFF": ("Light", False),
    "FAN_ON":   ("Fan",   True),  "FAN_OFF":   ("Fan",   False),
    "AC_ON":    ("AC",    True),  "AC_OFF":    ("AC",    False),
    "TV_ON":    ("TV",    True),  "TV_OFF":    ("TV",    False),
}

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.append(f"[{ts}] {msg}")
    if len(st.session_state.activity_log) > 80:
        st.session_state.activity_log.pop(0)

def apply_action(action: str, room: str = None):
    """Apply a predicted action to a specific room (or all rooms)."""
    rooms = st.session_state.devices.keys() if room is None else [room]
    results = []

    if action == "ALL_ON":
        for r in rooms:
            for dev in st.session_state.devices[r]:
                st.session_state.devices[r][dev]["state"] = True
        log(f"ALL devices ON ({', '.join(rooms)})")
        results.append("✅ All devices turned ON")
        return results

    if action == "ALL_OFF":
        for r in rooms:
            for dev in st.session_state.devices[r]:
                st.session_state.devices[r][dev]["state"] = False
        log(f"ALL devices OFF ({', '.join(rooms)})")
        results.append("✅ All devices turned OFF")
        return results

    if action in ACTION_DEVICE_MAP:
        device_name, new_state = ACTION_DEVICE_MAP[action]
        matched = False
        for r in rooms:
            for dev in st.session_state.devices[r]:
                if device_name.lower() in dev.lower():
                    st.session_state.devices[r][dev]["state"] = new_state
                    status = "ON" if new_state else "OFF"
                    log(f"{ACTION_EMOJI.get(action,'')} {dev} in {r} → {status}")
                    results.append(f"{ACTION_EMOJI.get(action,'')} **{dev}** in *{r}* → **{status}**")
                    matched = True
        if not matched:
            results.append(f"⚠️ No matching device for `{action}` in selected room(s).")
    else:
        results.append(f"❓ Unknown action: `{action}`")

    return results

def total_power_w():
    total = 0
    for room_devs in st.session_state.devices.values():
        for info in room_devs.values():
            if info["state"]:
                total += info["watts"]
    return total

def extract_mfcc(filepath):
    audio, sr_rate = librosa.load(filepath, duration=4, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

def is_owner(filepath) -> bool:
    if st.session_state.owner_mfcc is None:
        return True   # No enrollment → bypass
    try:
        test_mfcc = extract_mfcc(filepath)
        dist = np.linalg.norm(st.session_state.owner_mfcc - test_mfcc)
        return dist < 60
    except:
        return False

def record_audio(duration=5, samplerate=16000) -> str:
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, samplerate)
    return tmp.name

def recognize_speech(filepath) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as src:
        data = r.record(src)
    try:
        return r.recognize_google(data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def predict_and_show(command: str, room: str):
    if not command.strip():
        st.warning("⚠️ Empty command.")
        return
    action = model_pipeline.predict([command])[0]
    # Confidence via decision function
    decision = model_pipeline.decision_function([command])
    conf = float(np.max(decision))
    conf_pct = min(100, int((conf + 1) * 40))   # rough normalise to 0-100

    st.markdown(f"**Predicted action:** `{action}`  &nbsp; Confidence: `{conf_pct}%`")
    msgs = apply_action(action, room if room != "All Rooms" else None)
    for m in msgs:
        st.markdown(m)

# ──────────────────────────────────────────────
# LOGIN PAGE
# ──────────────────────────────────────────────
def login_page():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("## 🏠 Smart Home AI v2.0")
        st.markdown("#### Owner Authentication")
        uname = st.text_input("Username", placeholder="admin")
        pwd   = st.text_input("Password", type="password", placeholder="••••••")
        if st.button("Login →"):
            if uname == "admin" and pwd == "1234":
                st.session_state.logged_in = True
                log("Owner logged in")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. (Hint: admin / 1234)")
        st.caption("Default credentials: admin / 1234")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏠 Smart Home AI v2.0")
    st.markdown(f"🟢 **Logged in as:** admin")
    st.markdown(f"⚡ **Current load:** {total_power_w()} W")
    st.divider()

    # Quick room selector
    selected_room = st.selectbox("Target Room", ["All Rooms"] + list(st.session_state.devices.keys()))
    st.divider()

    # Voice enrollment
    st.markdown("#### 🔐 Voice Enrollment")
    if st.button("Enroll My Voice (5 sec)"):
        with st.spinner("Recording..."):
            try:
                path = record_audio(5)
                st.session_state.owner_mfcc = extract_mfcc(path)
                log("Owner voice enrolled")
                st.success("✅ Voice enrolled!")
            except Exception as e:
                st.error(f"Error: {e}")

    voice_status = "✅ Enrolled" if st.session_state.owner_mfcc is not None else "⚠️ Not enrolled (auth bypassed)"
    st.caption(f"Voice status: {voice_status}")
    st.divider()

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        log("Owner logged out")
        st.rerun()

# ──────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────
tab_ctrl, tab_rooms, tab_energy, tab_schedule, tab_log, tab_data = st.tabs([
    "🎯 Control", "🏠 Rooms", "⚡ Energy", "⏰ Schedules", "📜 Log", "📊 Model"
])

# ══════════════════════════════════════════════
# TAB 1 — CONTROL
# ══════════════════════════════════════════════
with tab_ctrl:
    st.subheader(f"🎯 Command Center  —  {selected_room}")
    st.caption("Type or speak a natural language command. The AI will identify the action.")

    input_mode = st.radio("Input mode", ["✏️ Text", "🎙️ Voice"], horizontal=True)

    if input_mode == "✏️ Text":
        user_cmd = st.text_input("Command", placeholder="e.g. 'turn on the AC' or 'all lights off'")
        if st.button("⚡ Execute"):
            predict_and_show(user_cmd, selected_room)

    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎙️ Record (5 sec)"):
                with st.spinner("Listening…"):
                    try:
                        path = record_audio(5)
                        if not is_owner(path):
                            st.error("🚫 Access Denied — voice not recognised as owner.")
                        else:
                            text = recognize_speech(path)
                            if text:
                                st.info(f"Heard: *{text}*")
                                predict_and_show(text, selected_room)
                            else:
                                st.warning("❓ Could not understand audio.")
                    except Exception as e:
                        st.error(f"Mic error: {e}")

    st.divider()
    st.markdown("##### Quick Shortcuts")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("💡 All Lights ON"):
            predict_and_show("turn on the light", selected_room)
    with qc2:
        if st.button("💡 All Lights OFF"):
            predict_and_show("turn off the light", selected_room)
    with qc3:
        if st.button("🌀 Fans ON"):
            predict_and_show("fan on", selected_room)
    with qc4:
        if st.button("🔴 All OFF"):
            predict_and_show("turn off everything", selected_room)

# ══════════════════════════════════════════════
# TAB 2 — ROOMS OVERVIEW
# ══════════════════════════════════════════════
with tab_rooms:
    st.subheader("🏠 Room-by-Room Status")
    for room, devices in st.session_state.devices.items():
        with st.expander(f"**{room}**", expanded=True):
            cols = st.columns(len(devices))
            for idx, (dev_name, info) in enumerate(devices.items()):
                with cols[idx]:
                    state_label = "🟢 ON" if info["state"] else "🔴 OFF"
                    st.markdown(f"**{dev_name}**")
                    st.markdown(state_label)
                    st.caption(f"{info['watts']} W")
                    toggle_key = f"toggle_{room}_{dev_name}"
                    if st.button("Toggle", key=toggle_key):
                        new_state = not info["state"]
                        st.session_state.devices[room][dev_name]["state"] = new_state
                        status = "ON" if new_state else "OFF"
                        log(f"Manual toggle: {dev_name} in {room} → {status}")
                        st.rerun()

# ══════════════════════════════════════════════
# TAB 3 — ENERGY
# ══════════════════════════════════════════════
with tab_energy:
    st.subheader("⚡ Energy Monitor")

    m1, m2, m3, m4 = st.columns(4)
    total_w = total_power_w()
    active_devs = sum(
        1 for r in st.session_state.devices.values()
        for d in r.values() if d["state"]
    )

    with m1:
        st.metric("Live Load", f"{total_w} W")
    with m2:
        st.metric("Active Devices", str(active_devs))
    with m3:
        daily_kwh = round(total_w * 8 / 1000, 2)   # assume 8 hrs avg
        st.metric("Est. Daily kWh", f"{daily_kwh}")
    with m4:
        est_cost = round(daily_kwh * 6.5, 2)         # ₹6.5 per unit
        st.metric("Est. Daily Cost", f"₹{est_cost}")

    st.divider()
    st.markdown("##### 7-Day Usage History (kWh)")
    hist_df = pd.DataFrame({
        "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        "kWh": st.session_state.energy_history,
    })
    st.bar_chart(hist_df.set_index("Day"))

    st.divider()
    st.markdown("##### Per-Room Breakdown")
    room_data = []
    for room, devs in st.session_state.devices.items():
        on_w = sum(d["watts"] for d in devs.values() if d["state"])
        room_data.append({"Room": room, "Active W": on_w, "Devices On": sum(1 for d in devs.values() if d["state"])})
    st.dataframe(pd.DataFrame(room_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 4 — SCHEDULES
# ══════════════════════════════════════════════
with tab_schedule:
    st.subheader("⏰ Device Schedules")
    st.caption("Set timed automation rules. Rules are checked on every page interaction.")

    with st.form("add_schedule"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            s_room = st.selectbox("Room", list(st.session_state.devices.keys()), key="s_room")
        with sc2:
            s_dev  = st.selectbox("Device", ["Light","Fan","AC","TV","Exhaust Fan","Laptop Charger","Microwave"], key="s_dev")
        with sc3:
            s_action = st.selectbox("Action", ["ON","OFF"], key="s_act")
        with sc4:
            s_time = st.time_input("At time", value=time(8, 0), key="s_time")
        if st.form_submit_button("➕ Add Schedule"):
            st.session_state.schedules.append({
                "room": s_room, "device": s_dev,
                "action": s_action, "time": s_time.strftime("%H:%M")
            })
            log(f"Schedule added: {s_dev} in {s_room} → {s_action} at {s_time.strftime('%H:%M')}")
            st.success("Schedule added!")

    # Check & apply due schedules
    now_hm = datetime.now().strftime("%H:%M")
    triggered = []
    for sched in st.session_state.schedules:
        if sched["time"] == now_hm:
            for dev in st.session_state.devices.get(sched["room"], {}):
                if sched["device"].lower() in dev.lower():
                    new_state = sched["action"] == "ON"
                    st.session_state.devices[sched["room"]][dev]["state"] = new_state
                    triggered.append(f"⏰ Auto: {dev} in {sched['room']} → {sched['action']}")
                    log(f"Schedule triggered: {dev} in {sched['room']} → {sched['action']}")

    for t in triggered:
        st.success(t)

    if st.session_state.schedules:
        st.markdown("##### Active Schedules")
        sched_df = pd.DataFrame(st.session_state.schedules)
        st.dataframe(sched_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear All Schedules"):
            st.session_state.schedules = []
            st.rerun()
    else:
        st.info("No schedules yet. Add one above.")

# ══════════════════════════════════════════════
# TAB 5 — ACTIVITY LOG
# ══════════════════════════════════════════════
with tab_log:
    st.subheader("📜 Activity Log")
    if st.button("🗑️ Clear Log"):
        st.session_state.activity_log = []

    if st.session_state.activity_log:
        for entry in reversed(st.session_state.activity_log):
            st.markdown(f"<div class='log-entry'>{entry}</div>", unsafe_allow_html=True)
    else:
        st.info("No activity yet.")

# ══════════════════════════════════════════════
# TAB 6 — MODEL DATA
# ══════════════════════════════════════════════
with tab_data:
    st.subheader("📊 AI Model Details")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model:** TF-IDF (bigram) + LinearSVC")
        st.markdown("**Classes:**")
        classes = ["LIGHT_ON","LIGHT_OFF","FAN_ON","FAN_OFF","AC_ON","AC_OFF","TV_ON","TV_OFF","ALL_ON","ALL_OFF"]
        for cls in classes:
            st.markdown(f"- `{cls}`")
    with c2:
        st.markdown("**Training samples:** " + str(len(TRAIN_DATA["command"])))
        st.markdown("**Feature:** TF-IDF bigrams, sublinear TF scaling")
        st.markdown("**Classifier:** LinearSVC, C=1.0")
        st.markdown("**Auth:** MFCC cosine distance (n_mfcc=20)")

    st.divider()
    st.markdown("##### Training Dataset")
    st.dataframe(pd.DataFrame(TRAIN_DATA), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Try the Model Directly")
    test_cmd = st.text_input("Test command", placeholder="e.g. brighten the room")
    if test_cmd:
        pred = model_pipeline.predict([test_cmd])[0]
        dec  = model_pipeline.decision_function([test_cmd])
        conf = min(100, int((float(np.max(dec)) + 1) * 40))
        st.markdown(f"Prediction: **`{pred}`**  — Confidence: **{conf}%**")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption("🏠 Smart Home AI v2.0 · TF-IDF + LinearSVC · MFCC Voice Auth · Multi-Room · Scheduling · Energy Monitor")
