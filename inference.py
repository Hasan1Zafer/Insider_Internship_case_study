from flask import Flask, request, jsonify
import joblib


model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the product description from the request
    data = request.get_json(force=True)
    description = data['description']


    processed_description = vectorizer.transform([description])
    prediction = model.predict(processed_description)


    response = {
        'description': description,
        'category': prediction[0]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
