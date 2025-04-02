import joblib

# Load the trained model
model = joblib.load("relapse_predictor.pkl")

# Custom input: [heart_rate, hrv, sleep_score, steps, temperature, spo2, stress]
#input_data = [[85, 30, 50, 2000, 37.5, 94, 85]]
input_data = [[72, 65, 88, 6500, 36.7, 97, 20]]
# Predict
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

# Output result
print(" Raw prediction:", prediction[0])
print(" Probability (Relapse):", round(probability[0][1] * 100, 2), "%")

if prediction[0] == 1:
    print("khouna seyes rohek  High chance of MS relapse.")
else:
    print("Jawek behy Stable condition. No relapse detected.")

