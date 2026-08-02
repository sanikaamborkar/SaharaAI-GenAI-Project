import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = r"E:\Gen-AI-Project\SahaaraAI\models\mentalbert-risk-classifier"
LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
COLORS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

st.set_page_config(page_title="MentalBERT Risk Classifier Test", layout="centered")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("🧠 MentalBERT Risk Classifier — Test Console")
st.caption("Standalone testing tool for the fine-tuned risk classification model (85% test accuracy)")

st.divider()

message = st.text_area(
    "Enter a message to classify:",
    placeholder="Type a sample message here...",
    height=100
)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Classifying..."):
            inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()
            predicted_id = torch.argmax(probs).item()
            predicted_label = LABELS[predicted_id]
            scores = {LABELS[i]: probs[i].item() for i in range(len(LABELS))}

        st.divider()
        st.markdown(f"### {COLORS[predicted_label]} Predicted: **{predicted_label}**")

        st.markdown("#### Confidence scores")
        for label in ["LOW", "MEDIUM", "HIGH"]:
            st.progress(scores[label], text=f"{COLORS[label]} {label}: {scores[label]*100:.1f}%")

st.divider()

st.markdown("#### Quick test examples")
st.caption("Click to auto-fill and classify")

examples = {
    "Low risk": "I had a great day at college today, feeling pretty good!",
    "Medium risk": "I feel really overwhelmed and can't cope with things lately.",
    "High risk": "I don't want to be here anymore and see no point in going on.",
}

cols = st.columns(3)
for col, (label, text) in zip(cols, examples.items()):
    if col.button(label):
        st.session_state["prefill"] = text
        st.rerun()

if "prefill" in st.session_state:
    st.info(f"Example loaded: \"{st.session_state['prefill']}\" — paste it into the box above and click Classify.")