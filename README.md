# Email Spam Checker

A local web app that learns to classify email text as spam or not spam. It uses public labeled email messages from the [Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/), instead of a few hand-written examples.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python download_data.py
python app.py
```

The dataset download is several megabytes. Leave the terminal open, then visit **http://127.0.0.1:5000** in your browser. Paste a message into the page to get a prediction. Press **Ctrl+C** in the terminal to stop the app.

If using VS Code, run these commands with the same Python interpreter selected in its bottom-right corner.

## How it works

- `download_data.py` downloads and extracts public spam and non-spam email archives.
- `SpamDetection.py` reads the email subject and plain-text body, then trains a TF-IDF and Naive Bayes model. It also reports accuracy on a held-out portion of the data.
- `app.py` provides a simple local web page for checking pasted messages.
- `main.py` runs the text-based version in the terminal.

The corpus is old and does not represent every kind of email today. The displayed accuracy is only a measurement on a held-out portion of this same corpus, not a guarantee on new mail. This app does not connect to an inbox or automatically move, delete, or send email. Treat predictions as suggestions, not proof that a message is safe or harmful.

The dataset is downloaded to `data/` and excluded from Git, so it is not included when you push this project. A person cloning the project needs to run `python download_data.py` before starting the app.
