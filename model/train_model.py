
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import mlflow

# Feature engineering
def create_features(df):
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['amount_log'] = np.log1p(df['amount'])
    return df

# MLflow tracking
mlflow.set_experiment("fraud_detection_v2")

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('fraud_data.csv')
    data = create_features(data)
    
    # Time-based split
    train = data[data['time'] < '2025-04-01 10:20:00']
    test = data[data['time'] >= '2025-04-01 10:20:00']
    
    # Pipeline
    preprocessor = ColumnTransformer([
        ('num', RobustScaler(), ['amount_log', 'hour']),
        ('cat', OneHotEncoder(), ['location'])
    ])
    
    model = make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_iter=200, 
            early_stopping=True,
            class_weight={0:1, 1:10}
        )
    )
    
    # Hyperparameter tuning
    param_dist = {
        'histgradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
        'histgradientboostingclassifier__max_depth': [3, 5, 7]
    }
    
    with mlflow.start_run():
        search = RandomizedSearchCV(
            model, param_dist, n_iter=10, cv=TimeSeriesSplit(3),
            scoring='average_precision'
        )
        search.fit(train.drop('fraud', axis=1), train['fraud'])
        
        # Evaluate
        probs = search.predict_proba(test)[:, 1]
        metrics = {
            'roc_auc': roc_auc_score(test['fraud'], probs),
            'pr_auc': average_precision_score(test['fraud'], probs)
        }
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(search.best_estimator_, "model")
        
        # Save model
        joblib.dump(search.best_estimator_, 'fraud_detection_model_v2.pkl')
