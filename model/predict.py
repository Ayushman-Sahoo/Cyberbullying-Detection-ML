import json
import re
from pathlib import Path

import joblib
import numpy as np


class CyberbullyingPredictor:
    def __init__(self, model_dir: Path | str | None = None):
        base = Path(__file__).resolve().parent
        self.model_dir = Path(model_dir) if model_dir else (base / "saved")
        package = joblib.load(self.model_dir / "cyberbullying_model.joblib")
        self.model = package["model"]
        self.vectorizer = package["vectorizer"]
        self.labels = package.get("labels", {0: "Not Harmful", 1: "Harmful"})
        self.model_name = package.get("model_name", "UnknownModel")
        self.harmful_threshold = float(package.get("harmful_threshold", 0.5))
        self.toxic_terms = (
            "idiot",
            "stupid",
            "useless",
            "worthless",
            "disgusting",
            "pathetic",
            "hate you",
            "shut up",
            "loser",
            "trash",
            "nobody wants you",
        )
        self.negation_patterns = (
            "not bad",
            "not stupid",
            "not useless",
            "not hate",
            "hate that",
        )

    def predict_text(self, text: str) -> dict:
        clean_text = str(text).strip()
        if not clean_text:
            raise ValueError("Text input cannot be empty.")

        X = self.vectorizer.transform([clean_text])
        pred = int(self.model.predict(X)[0])

        confidence = None
        class_probabilities: list[dict[str, float | str]] = []
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            confidence = float(np.max(probs))
            class_probabilities = [
                {"class": self.labels.get(int(idx), str(idx)), "probability": float(prob)}
                for idx, prob in enumerate(probs)
            ]
            class_probabilities.sort(key=lambda x: float(x["probability"]), reverse=True)
            harmful_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            pred = 1 if harmful_prob >= self.harmful_threshold else 0
            normalized = re.sub(r"\s+", " ", clean_text.lower()).strip()
            toxic_hit = any(term in normalized for term in self.toxic_terms)
            negated = any(pattern in normalized for pattern in self.negation_patterns)
            if toxic_hit and not negated and harmful_prob >= 0.45:
                pred = 1

        predicted_class = self.labels.get(pred, str(pred))
        message = (
            "Potential cyberbullying/toxic language detected. Please review this message."
            if pred == 1
            else "Message appears non-harmful."
        )
        return {
            "prediction": pred,
            "predicted_class": predicted_class,
            "model_name": self.model_name,
            "harmful_threshold": self.harmful_threshold,
            "confidence": confidence,
            "message": message,
            "class_probabilities": class_probabilities,
            "input_text": clean_text,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict cyberbullying/toxic content from text")
    parser.add_argument("--text", type=str, required=True, help="Text input to classify")
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()

    predictor = CyberbullyingPredictor(model_dir=args.model_dir)
    result = predictor.predict_text(args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

