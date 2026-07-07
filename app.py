from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(open("model/classifier.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

def assign_priority(text):
    if any(k in text for k in ["not working", "refund", "error", "failed"]):
        return "High"
    elif any(k in text for k in ["slow", "billing"]):
        return "Medium"
    return "Low"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticket = request.form["ticket"]

        cleaned = clean_text(ticket)
        vector = vectorizer.transform([cleaned])
        category = model.predict(vector)[0]
        priority = assign_priority(cleaned)

        return render_template(
            "result.html",
            ticket=ticket,
            category=category,
            priority=priority
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
