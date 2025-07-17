import streamlit as st
import torch
import torch.nn.functional as F
import re
import pickle
from model import TextCNN

# ------------------------
# 1. Load Saved Assets
# ------------------------

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

emotion_to_emoji = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐"
}

color_map = {
    "joy": "#FFD700",        # Gold
    "sadness": "#1E90FF",    # DodgerBlue
    "anger": "#FF4500",      # OrangeRed
    "fear": "#9370DB",       # MediumPurple
    "surprise": "#FFA500",   # Orange
    "love": "#FF69B4",       # HotPink
    "neutral": "#A9A9A9"      # DarkGray
}

# ------------------------
# 2. Preprocessing
# ------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def text_to_sequence(text, word2idx, max_length=50):
    tokens = text.split()
    sequence = [word2idx.get(word, word2idx['<OOV>']) for word in tokens]
    if len(sequence) < max_length:
        sequence += [word2idx['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return torch.tensor(sequence, dtype=torch.long).unsqueeze(0)

# ------------------------
# 3. Load Model
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(len(word2idx), 128, len(label_names), word2idx['<PAD>'])
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# ------------------------
# 4. Streamlit UI
# ------------------------

st.set_page_config(page_title="Emotion Detection", layout="centered", page_icon="🧠")
st.markdown("""
    <style>
        .main { background-color: #0d1117; color: white; }
        h1 { text-align: center; color: #FF69B4; }
        .stTextArea textarea { background-color: #1e1e1e; color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Emotion Detection from Text 🎉</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a sentence and the model will predict the emotion.</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter text here:", height=150)

if st.button("Predict Emotion ⚡") and user_input:
    cleaned = clean_text(user_input)
    input_tensor = text_to_sequence(cleaned, word2idx).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()
        pred_label = label_names[pred]
        emoji = emotion_to_emoji.get(pred_label, "")
        color = color_map.get(pred_label, "#333")

    st.markdown(f"""
        <div style='background-color:{color}; padding: 15px; border-radius:10px; text-align:center;'>
            <h3>Predicted Emotion: {pred_label.upper()} {emoji}</h3>
        </div>
    """, unsafe_allow_html=True)
