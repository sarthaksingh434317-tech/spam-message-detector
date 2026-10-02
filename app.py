from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            result = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            confidence = round(max(probabilities) * 100, 2)

            if result == 1:
                prediction = "SPAM"
            else:
                prediction = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)