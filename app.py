"""Local web interface for the email spam classifier."""

from flask import Flask, render_template, request

from SpamDetection import clean_input, train_model


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
model, stats = train_model()


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    message = ""
    error = None
    if request.method == "POST":
        message = request.form.get("message", "")[:100_000]
        cleaned = clean_input(message)
        if not cleaned:
            error = "Paste an email message first."
        else:
            prediction = int(model.predict([cleaned])[0])
            result = "Spam" if prediction == 1 else "Not spam"
    return render_template(
        "index.html", result=result, message=message, error=error, stats=stats
    )


if __name__ == "__main__":
    # Local development only. Do not expose Flask's debug server to the internet.
    app.run(host="127.0.0.1", port=5000, debug=False)
