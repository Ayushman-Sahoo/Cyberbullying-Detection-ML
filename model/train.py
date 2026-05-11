import argparse
import glob
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def _collect_csv_paths(raw_value: str) -> list[Path]:
    items = [x.strip().strip('"').strip("'") for x in raw_value.split(",") if x.strip()]
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_file():
            paths.append(p)
            continue
        if p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
            continue
        if "*" in item or "?" in item:
            paths.extend(Path(x) for x in glob.glob(item) if Path(x).is_file())
            continue
        raise FileNotFoundError(f"Path not found: {item}")
    unique: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve()).lower()
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def _get_dataset_paths(cli_data_path: str | None) -> list[Path]:
    if cli_data_path:
        out = _collect_csv_paths(cli_data_path)
        if not out:
            raise ValueError("No CSV files found from --data-path.")
        return out
    user_input = input("Enter dataset file path(s), folder path(s), or CSV wildcard (comma-separated): ").strip()
    if not user_input:
        raise ValueError("No dataset path provided.")
    out = _collect_csv_paths(user_input)
    if not out:
        raise ValueError("No CSV files found from provided input.")
    return out


def _find_text_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ["tweet", "comment_text", "text", "comment", "content", "message"]:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("Could not detect text column. Expected one of: tweet, comment_text, text, comment, content, message.")


def _find_label_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ["class", "label", "target", "outcome"]:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("Could not detect label column. Expected one of: class, label, target, outcome.")


def _map_label(value: object, source_name: str) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    try:
        numeric = float(text)
        if source_name == "class":
            return 0 if int(numeric) == 2 else 1
        return 1 if numeric > 0 else 0
    except ValueError:
        pass

    if text in {"not", "non-toxic", "nontoxic", "clean", "safe", "neutral", "neither", "normal", "0"}:
        return 0
    if any(token in text for token in ["toxic", "hate", "offensive", "bully", "abuse", "insult", "threat", "harass"]):
        return 1
    return None


def _metrics(y_true: pd.Series, y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _build_safety_augmentation(repeats: int = 15) -> pd.DataFrame:
    safe_phrases = [
        "You are brilliant and your effort is inspiring.",
        "You are amazing and I appreciate your kindness.",
        "You are not bad at all, you did really well.",
        "You are not stupid, your idea is thoughtful.",
        "You are crazy talented and very creative.",
        "Your presentation was killer in a good way.",
        "That was a sick performance, excellent work.",
        "I hate that this event ended, it was so good.",
        "You are too good at this task, great job.",
        "Thank you for helping me, you are the best.",
        "Your feedback was clear, polite, and useful.",
        "I am proud of your progress and dedication.",
        "You handled the situation with maturity and respect.",
        "Great work, your contribution made this project better.",
        "You are a strong teammate and very supportive.",
    ]
    harmful_phrases = [
        "You are useless and everyone hates you.",
        "Shut up, nobody wants to hear from you.",
        "You are disgusting and pathetic.",
        "I will hurt you if you speak again.",
        "You are a loser and should disappear.",
        "You are trash and your life is worthless.",
        "Go away, you are not wanted here.",
        "I will make you regret this.",
    ]

    rows: list[dict[str, int | str]] = []
    for _ in range(max(1, repeats)):
        rows.extend({"text": text, "label": 0} for text in safe_phrases)
    for _ in range(max(1, repeats // 2)):
        rows.extend({"text": text, "label": 1} for text in harmful_phrases)
    return pd.DataFrame(rows)


def _choose_harmful_threshold(y_true: pd.Series, harmful_prob: pd.Series) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_stats = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for step in range(50, 71):
        threshold = step / 100.0
        pred = (harmful_prob >= threshold).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if recall < 0.75:
            continue
        if precision > best_stats["precision"] or (
            precision == best_stats["precision"] and f1 > best_stats["f1"]
        ):
            best_threshold = threshold
            best_stats = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cyberbullying Detection model")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset path(s), folder path(s), or wildcard")
    parser.add_argument("--model-dir", type=str, default=str(Path(__file__).resolve().parent / "saved"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    dataset_paths = _get_dataset_paths(args.data_path)
    print(f"Using {len(dataset_paths)} dataset file(s):")
    for p in dataset_paths:
        print(f"- {p}")

    rows: list[pd.DataFrame] = []
    for p in dataset_paths:
        loaded = pd.read_csv(p)
        if loaded.empty:
            continue
        text_col = _find_text_column(loaded)
        label_col = _find_label_column(loaded)
        subset = loaded[[text_col, label_col]].copy()
        subset.columns = ["text", "label_raw"]
        subset["label"] = subset["label_raw"].apply(lambda x: _map_label(x, str(label_col).strip().lower()))
        subset = subset.dropna(subset=["text", "label"])
        subset["text"] = subset["text"].astype(str).str.strip()
        subset = subset[subset["text"] != ""]
        subset["label"] = subset["label"].astype(int)
        if not subset.empty:
            rows.append(subset[["text", "label"]])

    if not rows:
        raise ValueError("No usable rows were loaded from provided datasets.")

    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    augmented = _build_safety_augmentation(repeats=15)
    df = pd.concat([df, augmented], ignore_index=True).drop_duplicates()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"] != ""]
    if df["label"].nunique() < 2:
        raise ValueError("Training data must contain both harmful and non-harmful labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results: dict[str, dict] = {}

    print("\nTraining MultinomialNB...")
    nb = MultinomialNB(alpha=0.2)
    nb.fit(X_train_vec, y_train)
    nb_pred = nb.predict(X_test_vec)
    nb_metrics = _metrics(y_test, nb_pred.tolist())
    print(f"MultinomialNB f1: {nb_metrics['f1']:.4f}")
    results["MultinomialNB"] = {"metrics": nb_metrics, "model": nb, "pred": nb_pred}

    print("\nTraining LogisticRegression...")
    lr = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=args.random_state,
    )
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)
    lr_metrics = _metrics(y_test, lr_pred.tolist())
    print(f"LogisticRegression f1: {lr_metrics['f1']:.4f}")
    results["LogisticRegression"] = {"metrics": lr_metrics, "model": lr, "pred": lr_pred}

    best_name = max(results.keys(), key=lambda n: results[n]["metrics"]["f1"])
    best = results[best_name]
    harmful_threshold = 0.5
    threshold_metrics = None
    if hasattr(best["model"], "predict_proba"):
        best_probs = best["model"].predict_proba(X_test_vec)[:, 1]
        harmful_threshold, threshold_metrics = _choose_harmful_threshold(y_test, pd.Series(best_probs))
    best_report = classification_report(y_test, best["pred"], target_names=["Not Harmful", "Harmful"], zero_division=0)

    package = {
        "model_name": best_name,
        "model": best["model"],
        "vectorizer": vectorizer,
        "labels": {0: "Not Harmful", 1: "Harmful"},
        "harmful_threshold": harmful_threshold,
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_dir / "cyberbullying_model.joblib", compress=3)

    metadata = {
        "best_model": best_name,
        "metrics": best["metrics"],
        "all_model_metrics": {k: v["metrics"] for k, v in results.items()},
        "num_rows": int(len(df)),
        "harmful_ratio": float(df["label"].mean()),
        "augmentation_rows": int(len(augmented)),
        "harmful_threshold": harmful_threshold,
        "threshold_metrics": threshold_metrics,
        "data_paths": [str(p) for p in dataset_paths],
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(best_report)

    print("\nSaved artifacts:")
    print(f"- {model_dir / 'cyberbullying_model.joblib'}")
    print(f"- {model_dir / 'metadata.json'}")
    print(f"- {model_dir / 'classification_report.txt'}")
    print(f"Best model: {best_name} | F1: {best['metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()

