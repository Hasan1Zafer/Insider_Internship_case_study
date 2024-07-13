## THİS FİLE FOR LOCALLY TESTİNG MODEL

import joblib


model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Test input description
description = "Yazıcı kartuşu ve toner"


processed_description = vectorizer.transform([description])

# Make a prediction
prediction = model.predict(processed_description)

print(f"Description: {description}")
print(f"Predicted Category: {prediction[0]}")
