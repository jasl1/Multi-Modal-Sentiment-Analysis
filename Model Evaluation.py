from sklearn.metrics import accuracy_score, classification_report

# Evaluate model performance
predictions = model.predict_classes(padded_sequences)
accuracy = accuracy_score(data['encoded_sentiment'], predictions)
report = classification_report(data['encoded_sentiment'], predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
