
import json
import joblib
import numpy as np
import logging
from schema import Schema, And, Use, SchemaError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load model
try:
    model = joblib.load('fraud_detection_model.pkl')
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

# Input validation schema
input_schema = Schema({
    'amount': And(Use(float), lambda n: n > 0),
    'time': And(str, Use(lambda s: pd.to_datetime(s))),
    'location': And(str, Use(str.lower)),
    'user_id': And(str, len),
    'transaction_type': And(str, Use(str.upper), lambda s: s in ['POS', 'ONLINE', 'ATM'])
})

def lambda_handler(event, context):
    try:
        # Validate input
        body = json.loads(event['body'])
        input_schema.validate(body)
        
        # Feature engineering
        features = preprocess_features(body)
        
        # Predict
        prediction = model.predict_proba([features])[0][1]
        
        # Audit log
        log_transaction(body, prediction)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'fraud_probability': float(prediction),
                'fraud_flag': bool(prediction > 0.85),
                'model_version': '1.2.0'
            })
        }
    
    except SchemaError as e:
        logger.warning(f"Invalid input: {str(e)}")
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid input format'})}
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': 'Internal server error'})}

def preprocess_features(data):
    return [
        data['amount'],
        pd.to_datetime(data['time']).hour,
        geocode_location(data['location']),
        transaction_frequency(data['user_id'])
    ]

def log_transaction(data, score):
    logger.info(json.dumps({
        **data,
        'fraud_score': score,
        'timestamp': datetime.utcnow().isoformat()
    }))
