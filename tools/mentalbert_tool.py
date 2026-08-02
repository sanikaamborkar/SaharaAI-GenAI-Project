"""
tools/mentalbert_tool.py


this tool does NOT decide final risk level on its own. Its
job is to hand Planner the full probability distribution plus explicit
uncertainty flags, because at 86% accuracy, trusting only the top
predicted label silently hides near-tie cases (e.g. LOW=45.8% vs
HIGH=45.0%) where the model is genuinely unsure. Planner (an LLM reading
the raw text) is much better equipped to resolve ambiguity/negation than
this classifier is, so Screener's job is to surface uncertainty, not
hide it behind a single confident-looking label.
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crewai.tools import tool


# CONFIG

MODEL_PATH = os.path.join("models", "mentalbert-risk-classifier")
LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# If the top two scores are closer than this, the prediction is flagged
# as uncertain rather than trusted at face value.
UNCERTAINTY_MARGIN = 0.15

print(f"Loading fine-tuned risk classifier from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"MentalBERT loaded on device: {device}")


def _classify(message: str) -> dict:
    """Core inference — returns predicted label, full scores, and uncertainty flags."""
    inputs = tokenizer(
        message, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).squeeze()
    predicted_label = LABELS[torch.argmax(probs).item()]
    scores = {LABELS[i]: round(probs[i].item(), 3) for i in range(len(LABELS))}

    sorted_vals = sorted(scores.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    uncertain = margin < UNCERTAINTY_MARGIN

    # Near-HIGH flag: even if HIGH didn't "win", flag if it came close to
    # the top score. This is the safety-critical case — a message that's
    # predicted LOW/MEDIUM but where HIGH was nearly tied should NOT be
    # treated as confidently safe.
    high_score = scores["HIGH"]
    top_score = sorted_vals[0]
    near_high_risk = (predicted_label != "HIGH") and (top_score - high_score < UNCERTAINTY_MARGIN)

    return {
        "predicted_label": predicted_label,
        "scores": scores,
        "margin": round(margin, 3),
        "uncertain": uncertain,
        "near_high_risk": near_high_risk,
    }


@tool("mentalbert_screen")
def mentalbert_screen(message: str) -> str:
    """
    Screens a user's message for distress risk level using a fine-tuned
    MentalBERT classifier (~86% test accuracy). Always use this tool first,
    on every incoming user message, before any other reasoning. Returns the
    predicted label, full score distribution, and explicit uncertainty
    flags — do not trust the predicted label alone if uncertain=True or
    near_high_risk=True is present in the output.
    """
    try:
        result = _classify(message)

        warnings = []
        if result["uncertain"]:
            warnings.append(
                f"UNCERTAIN — top two categories nearly tied (margin="
                f"{result['margin']}). Do not treat the predicted label as confident."
            )
        if result["near_high_risk"]:
            warnings.append(
                "CAUTION — HIGH risk score is close to the top score even "
                "though it wasn't the top prediction. Do not assume low risk "
                "based on the predicted label alone."
            )

        warning_text = " " + " ".join(warnings) if warnings else ""

        return (
            f"distress_level={result['predicted_label']}, "
            f"scores={result['scores']}, "
            f"uncertain={result['uncertain']}, "
            f"near_high_risk={result['near_high_risk']}."
            f"{warning_text}"
        )
    except Exception as e:
        return f"ERROR: MentalBERT screening failed: {e}"



# Standalone test — run from project root: python -m tools.mentalbert_tool

if __name__ == "__main__":
    test_messages = [
        "I had a great day at college today!",
        "I feel really overwhelmed and can't cope with things lately.",
        "I don't want to be here anymore.",
        "I keep thinking there's no point but I also don't want to die.",
    ]
    for msg in test_messages:
        print(f"\n'{msg}'\n  -> {mentalbert_screen.func(msg)}")
