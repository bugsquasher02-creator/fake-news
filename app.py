from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    data = vectorizer.transform([news])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
