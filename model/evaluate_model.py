
import joblib
from sklearn.metrics import roc_auc_score

# Load the model
model = joblib.load('fraud_detection_model.pkl')

# Evaluate the model using test data
def evaluate_model(test_data, true_labels):
    predictions = model.predict(test_data)
    auc_score = roc_auc_score(true_labels, predictions)
    print(f"Model AUC: {auc_score}")
