from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['news_text']
        user_input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vectorized)[0]
        result = "Fake News" if prediction == 1 else "Real News"
        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
