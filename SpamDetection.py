"""Train and run an email spam classifier using the SpamAssassin corpus."""

from email import policy
from email.parser import BytesParser
from html import unescape
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def email_to_text(path):
    """Read an email file and return its subject plus readable body text."""
    try:
        message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    except (OSError, ValueError):
        return ""

    parts = []
    subject = str(message.get("subject", ""))
    if subject:
        parts.append(subject)

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                try:
                    parts.append(part.get_content())
                except (LookupError, UnicodeError):
                    continue
    elif message.get_content_type() == "text/plain":
        try:
            parts.append(message.get_content())
        except (LookupError, UnicodeError):
            pass

    return " ".join(str(part) for part in parts if part).strip()


def load_dataset():
    """Load labeled messages from the downloaded SpamAssassin mail folders."""
    messages, labels = [], []
    for folder in DATA_DIR.iterdir() if DATA_DIR.exists() else []:
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if "spam" in name:
            label = 1
        elif "ham" in name:
            label = 0
        else:
            continue

        for path in folder.iterdir():
            if path.is_file() and not path.name.startswith("."):
                text = email_to_text(path)
                if text:
                    messages.append(text)
                    labels.append(label)

    if not messages or len(set(labels)) < 2:
        raise FileNotFoundError(
            "Email data is missing. Run 'python download_data.py' first."
        )
    return messages, labels


def create_model():
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        ),
        MultinomialNB(),
    )


def train_model():
    """Evaluate on a held-out sample, then fit the final model on all mail."""
    messages, labels = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    evaluation_model = create_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["Not spam", "Spam"], zero_division=0
    )

    model = create_model()
    model.fit(messages, labels)
    return model, {"total": len(messages), "accuracy": accuracy, "report": report}


def clean_input(text):
    """Keep user input plain text and remove markup-like tags."""
    text = unescape(re.sub(r"<[^>]*>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


def main():
    model, stats = train_model()
    print(f"Trained on {stats['total']:,} real email messages.")
    print(f"Held-out accuracy: {stats['accuracy']:.1%}\n")
    while True:
        try:
            text = input("Paste an email (or type 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            print("Please paste an email message.\n")
            continue
        result = model.predict([clean_input(text)])[0]
        print("SPAM\n" if result == 1 else "NOT SPAM\n")


if __name__ == "__main__":
    main()
