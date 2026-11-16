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
    # Listen on the local network so a phone on the same Wi-Fi can open the app.
    # This is still a development server; do not expose it to the public internet.
    app.run(host="0.0.0.0", port=5000, debug=False)
