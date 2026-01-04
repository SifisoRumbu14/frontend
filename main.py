from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator
from typing import List, Dict, Any
import torch
import tensorflow as tf
import numpy as np
import joblib
import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ------------------------------
# Initialize FastAPI App
# ------------------------------
app = FastAPI(title="Heart Attack Prediction API")

from fastapi.middleware.cors import CORSMiddleware

# Add this right after: app = FastAPI(title="Heart Attack Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------
# Pydantic Request Model
# ------------------------------
class PredictRequest(BaseModel):
    features: Dict[str, Any]
    framework: str  # "pytorch" or "tensorflow"

    @validator("framework")
    def validate_framework(cls, v):
        v = v.lower()
        if v not in ["pytorch", "tensorflow"]:
            raise ValueError(f"Unsupported framework: {v}")
        return v

# ------------------------------
# Feature Configuration
# ------------------------------
FEATURE_ORDER = [
    'age', 'gender', 'region', 'income_level', 'hypertension', 'diabetes',
    'cholesterol_level', 'obesity', 'waist_circumference', 'family_history',
    'smoking_status', 'physical_activity', 'dietary_habits',
    'air_pollution_exposure', 'stress_level', 'sleep_hours',
    'blood_pressure_systolic', 'blood_pressure_diastolic',
    'fasting_blood_sugar', 'cholesterol_hdl', 'cholesterol_ldl',
    'triglycerides', 'EKG_results', 'previous_heart_disease',
    'medication_usage', 'participated_in_free_screening'
]
CATEGORICAL_FEATURES = {
    'gender': ['Female', 'Male'],
    'region': ['Rural', 'Urban'], 
    'income_level': ['High', 'Low', 'Medium'],
    'smoking_status': ['Current','Never', 'Past'],
    'physical_activity': ['High', 'Low', 'Moderate'],
    'dietary_habits': ['Healthy', 'Unhealthy'],
    'air_pollution_exposure': ['High', 'Low', 'Medium'],
    'stress_level': ['High', 'Low', 'Medium'],
    'EKG_results': ['Abnormal', 'Normal']
  }

# ------------------------------
# Device Detection
# ------------------------------
def detect_device_type(user_agent: str) -> str:
    """Detect if the request is from a mobile device or desktop"""
    if not user_agent:
        return "desktop"
    
    mobile_patterns = [
        'mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry',
        'webos', 'opera mini', 'iemobile', 'windows phone'
    ]
    
    user_agent_lower = user_agent.lower()
    for pattern in mobile_patterns:
        if pattern in user_agent_lower:
            return "mobile"
    
    return "desktop"

# ------------------------------
# Model Loading - Desktop and Mobile (FIXED)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load encoders and scaler (same for both desktop and mobile)
label_encoders = {}
for feature, categories in CATEGORICAL_FEATURES.items():
    try:
        encoder_path = os.path.join(BASE_DIR, f"{feature}_encoder.pkl")
        if os.path.exists(encoder_path):
            label_encoders[feature] = joblib.load(encoder_path)
        else:
            encoder = LabelEncoder()
            encoder.fit(categories)
            label_encoders[feature] = encoder
    except Exception as e:
        logger.warning(f"Could not load encoder for {feature}: {e}")

scaler = None
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        logger.info("✅ Scaler loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load scaler: {e}")

# Load all model types
models_info = {
    'desktop': {
        'pytorch': {},
        'tensorflow': {}
    },
    'mobile': {
        'pytorch': {},
        'tensorflow': {}
    }
}

# Load Desktop PyTorch models
try:
    desktop_pytorch_models = {
        'mlp': torch.jit.load(os.path.join(BASE_DIR, "mlp_model.pt")),
        'cnn': torch.jit.load(os.path.join(BASE_DIR, "cnn_model.pt")),
        'gru': torch.jit.load(os.path.join(BASE_DIR, "gru_model.pt"))
    }
    
    for name, model in desktop_pytorch_models.items():
        model.eval()
        models_info['desktop']['pytorch'][name] = {
            'model': model,
            'expected_features': 31,
            'type': 'desktop'
        }
        logger.info(f"✅ Desktop PyTorch {name.upper()} loaded")
            
except Exception as e:
    logger.error(f"Failed to load desktop PyTorch models: {e}")

# Load Mobile PyTorch models (.ptl files) - with fallback
try:
    mobile_pytorch_files = {
        'mlp': "mlp_model_mobile.ptl",
        'cnn': "cnn_model_mobile.ptl", 
        'gru': "gru_model_mobile.ptl"
    }

    mobile_pytorch_models = {}
    for name, filename in mobile_pytorch_files.items():
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(file_path):
            mobile_pytorch_models[name] = torch.jit.load(file_path)
            mobile_pytorch_models[name].eval()
            models_info['mobile']['pytorch'][name] = {
                'model': mobile_pytorch_models[name],
                'expected_features': 31,
                'type': 'mobile'
            }
            logger.info(f"✅ Mobile PyTorch {name.upper()} loaded")
        else:
            logger.warning(f"⚠️ Mobile PyTorch {name} file not found: {filename}")
            
except Exception as e:
    logger.warning(f"Failed to load mobile PyTorch models: {e}")

# Load Desktop TensorFlow models
try:
    desktop_tf_files = {
        'mlp': "MLP-Full.keras",
        'cnn': "CNN-Full.keras",
        'gru': "GRU-Full.keras"
    }
    
    desktop_tf_models = {}
    for name, filename in desktop_tf_files.items():
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(file_path):
            desktop_tf_models[name] = tf.keras.models.load_model(file_path)
            models_info['desktop']['tensorflow'][name] = {
                'model': desktop_tf_models[name],
                'expected_features': 26,
                'type': 'desktop'
            }
            logger.info(f"✅ Desktop TensorFlow {name.upper()} loaded")
        else:
            logger.warning(f"⚠️ Desktop TensorFlow {name} file not found: {filename}")
            
except Exception as e:
    logger.error(f"Failed to load desktop TensorFlow models: {e}")

# Load Mobile TensorFlow Lite models - WITH BETTER ERROR HANDLING
try:
    # TensorFlow Lite interpreter setup
    def load_tflite_model(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    # Try multiple possible TFLite filenames
    tflite_filename_variants = {
        'mlp': ["MLP-Lite.tflite", "mlp_model.tflite", "mlp_mobile.tflite"],
        'cnn': ["CNN-Lite.tflite", "cnn_model.tflite", "cnn_mobile.tflite"],
        'gru': ["GRU-Lite.tflite", "gru_model.tflite", "gru_mobile.tflite"]
    }
    
    mobile_tflite_models = {}
    for model_type, filename_list in tflite_filename_variants.items():
        model_loaded = False
        for filename in filename_list:
            file_path = os.path.join(BASE_DIR, filename)
            if os.path.exists(file_path):
                try:
                    mobile_tflite_models[model_type] = load_tflite_model(file_path)
                    models_info['mobile']['tensorflow'][model_type] = {
                        'model': mobile_tflite_models[model_type],
                        'expected_features': 26,
                        'type': 'tflite'
                    }
                    logger.info(f"✅ Mobile TensorFlow Lite {model_type.upper()} loaded: {filename}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    continue
        
        if not model_loaded:
            logger.warning(f"⚠️ No TFLite model found for {model_type}. Tried: {filename_list}")
    
    # If no TFLite models found, fallback to using desktop models for mobile
    if not models_info['mobile']['tensorflow']:
        logger.info("🔄 No TFLite models found, using desktop TensorFlow models for mobile as fallback")
        for name in desktop_tf_files.keys():
            if name in desktop_tf_models:
                models_info['mobile']['tensorflow'][name] = {
                    'model': desktop_tf_models[name],
                    'expected_features': 26,
                    'type': 'desktop_fallback'
                }
                logger.info(f"🔄 Using desktop TensorFlow {name.upper()} as mobile fallback")
            
except Exception as e:
    logger.error(f"Failed to load mobile TensorFlow models: {e}")

# Log final model status
logger.info("📊 Final Model Loading Status:")
for device_type in ['desktop', 'mobile']:
    for framework in ['pytorch', 'tensorflow']:
        count = len(models_info[device_type][framework])
        logger.info(f"  {device_type.upper()} {framework.upper()}: {count} models")

# ------------------------------
# Feature Processing
# ------------------------------
def preprocess_features(features: Dict[str, Any], target_size: int = 26) -> np.ndarray:
    """Preprocess features: encode -> scale -> pad/truncate to target_size"""
    X = pd.DataFrame(features, index=[0])
    processed_features = []

    for col in X.columns:
        if col in CATEGORICAL_FEATURES.keys():
            try:
                processed_features.append(CATEGORICAL_FEATURES[col].index(X[col][0]) )
            except Exception as ex:
                print(f"Encoding Error: {ex}")
        else:
            processed_features.append(X.iloc[0, X.columns.get_loc(col)].item())
    #display(processed_features)
        
    features_array = np.array(processed_features, dtype=np.float32).reshape(1, -1)

    #print("Features BEFORE Scaling: ")      
    #display(features_array)

    ########### SCALAR ##################

    scaler = joblib.load('scaler2.pkl')
    if scaler is not None:
        features_array = scaler.transform(features_array)
        #pass

    #print(f"Features AFTER Scaling: \n{features_array}") 

    return features_array
# ------------------------------
# Prediction Functions for Different Model Types
# ------------------------------
def pytorch_predict(features_dict: Dict[str, Any], model_type: str, device_type: str):
    """Make prediction with PyTorch models (desktop or mobile)"""
    if device_type not in models_info or model_type not in models_info[device_type]['pytorch']:
        raise ValueError(f"PyTorch {model_type} model not loaded for {device_type}")
    
    model_info = models_info[device_type]['pytorch'][model_type]
    model = model_info['model']
    expected_features = model_info['expected_features']
    
    features_array = preprocess_features(features_dict, target_size=expected_features)
    
     #for Pytorch
    model.eval()
    # convert np_array to tensor
    X_tensor = torch.from_numpy(features_array).float()

    if 'GRU' in model.original_name:
      X_tensor = X_tensor.unsqueeze(2)

    with torch.no_grad():
      pred = model(X_tensor)
    print(f"Prediction is {pred}")
    return pred.item()

def tensorflow_predict(features_dict: Dict[str, Any], model_type: str, device_type: str):
    """Make prediction with TensorFlow models (desktop or mobile)"""
    if device_type not in models_info or model_type not in models_info[device_type]['tensorflow']:
        raise ValueError(f"TensorFlow {model_type} model not loaded for {device_type}")
    
    model_info = models_info[device_type]['tensorflow'][model_type]
    expected_features = model_info['expected_features']
        # Remove alcohol_consumption if present
    features_dict.pop('alcohol_consumption', None)
    print(features_dict)
    features_array = preprocess_features(features_dict, target_size=expected_features)
    print(features_array)
    if model_info['type'] == 'tflite':
        # TensorFlow Lite prediction
        interpreter = model_info['model']
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Reshape for different model types
        if model_type == "mlp":
            input_data = features_array.astype(np.float32)
        elif model_type == "cnn":
            input_data = features_array.reshape(1, features_array.shape[1], 1).astype(np.float32)
        elif model_type == "gru":
            input_data = features_array.astype(np.float32)  # Shape: [1, 26] - 2D
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = float(output_data[0][0])
    else:
        # Standard TensorFlow prediction (desktop or fallback)
        if model_type == "mlp":
            x_input = features_array
        elif model_type == "cnn":
            x_input = features_array.reshape(1, features_array.shape[1], 1)
        elif model_type == "gru":
            x_input = features_array.reshape(1, features_array.shape[1], 1)
        
        output = model_info['model'].predict(x_input, verbose=0)
        prediction = float(output[0][0])
    
    return prediction

def ensemble_predict(features_dict: Dict[str, Any], framework: str, device_type: str):
    """Make ensemble prediction using all available models"""
    predictions = []
    individual_predictions = {}
    successful_models = []
    
    for model_type in ['mlp', 'cnn', 'gru']:
        try:
            if framework == 'pytorch':
                pred = pytorch_predict(features_dict, model_type, device_type)
            else:  # tensorflow
                pred = tensorflow_predict(features_dict, model_type, device_type)
            
            predictions.append(pred)
            individual_predictions[model_type] = pred
            successful_models.append(model_type)
            logger.info(f"✅ {device_type.upper()} {framework.upper()} {model_type.upper()}: {pred:.6f}")
            
        except Exception as e:
            logger.warning(f"❌ {device_type.upper()} {framework.upper()} {model_type} failed: {e}")
            individual_predictions[model_type] = None
    
    if not predictions:
        raise ValueError(f"All {device_type} {framework} models failed")
    
    ensemble_prediction = sum(predictions) / len(predictions)
    logger.info(f"📊 {device_type.upper()} {framework.upper()} Ensemble: {ensemble_prediction:.6f}")
    
    return {
        "ensemble_prediction": ensemble_prediction,
        "individual_predictions": individual_predictions,
        "models_used": successful_models,
        "failed_models": [name for name in ['mlp', 'cnn', 'gru'] if name not in successful_models]
    }

# ------------------------------
# FastAPI Endpoints
# ------------------------------
@app.post("/predict")
async def predict(request: PredictRequest, fastapi_request: Request):
    """Main prediction endpoint with automatic device detection"""
    features_dict = request.features
    framework = request.framework
    
    # Detect device type from User-Agent header
    user_agent = fastapi_request.headers.get('user-agent', '')
    device_type = detect_device_type(user_agent)
    
    logger.info(f"🚀 Prediction request - Device: {device_type}, Framework: {framework}")

    try:
        # Validate features
        missing_features = set(FEATURE_ORDER) - set(features_dict.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Get ensemble prediction for detected device type
        ensemble_result = ensemble_predict(features_dict, framework, device_type)
        prediction = ensemble_result["ensemble_prediction"]

        # Create response
        interpretation = "Heart attack risk detected" if prediction >= 0.5 else "Low heart attack risk"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        result = {
            "framework": framework,
            "device_type": device_type,
            "prediction": float(f"{prediction:.6f}"),
            "interpretation": interpretation,
            "confidence": float(f"{confidence:.6f}"),
            "binary_classification": 1 if prediction >= 0.5 else 0,
            "model_type": f"{device_type}_ensemble",
            "individual_predictions": ensemble_result["individual_predictions"],
            "models_used": ensemble_result["models_used"],
            "failed_models": ensemble_result["failed_models"],
            "features_used": 31 if framework == "pytorch" else 26,
            "status": "success",
            "note": f"Using {device_type} models - ensemble of {len(ensemble_result['models_used'])} models"
        }

        logger.info(f"✅ {device_type.upper()} prediction successful")
        return result

    except Exception as e:
        logger.error(f"❌ {device_type.upper()} prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/force-desktop")
async def predict_force_desktop(request: PredictRequest):
    """Force using desktop models (for testing)"""
    features_dict = request.features
    framework = request.framework

    logger.info(f"🖥️ Forced desktop prediction - Framework: {framework}")

    try:
        ensemble_result = ensemble_predict(features_dict, framework, "desktop")
        prediction = ensemble_result["ensemble_prediction"]

        interpretation = "Heart attack risk detected" if prediction >= 0.5 else "No heart attack risk"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        return {
            "framework": framework,
            "device_type": "desktop",
            "prediction": float(f"{prediction:.6f}"),
            "interpretation": interpretation,
            "confidence": float(f"{confidence:.6f}"),
            "binary_classification": 1 if prediction >= 0.5 else 0,
            "model_type": "desktop_ensemble",
            "individual_predictions": ensemble_result["individual_predictions"],
            "models_used": ensemble_result["models_used"],
            "features_used": 31 if framework == "pytorch" else 26,
            "status": "success",
            "note": "Forced desktop models"
        }

    except Exception as e:
        logger.error(f"❌ Forced desktop prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/force-mobile")
async def predict_force_mobile(request: PredictRequest):
    """Force using mobile models (for testing)"""
    features_dict = request.features
    framework = request.framework

    logger.info(f"📱 Forced mobile prediction - Framework: {framework}")

    try:
        ensemble_result = ensemble_predict(features_dict, framework, "mobile")
        prediction = ensemble_result["ensemble_prediction"]

        interpretation = "Heart attack risk detected" if prediction >= 0.5 else "No heart attack risk"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        return {
            "framework": framework,
            "device_type": "mobile",
            "prediction": float(f"{prediction:.6f}"),
            "interpretation": interpretation,
            "confidence": float(f"{confidence:.6f}"),
            "binary_classification": 1 if prediction >= 0.5 else 0,
            "model_type": "mobile_ensemble",
            "individual_predictions": ensemble_result["individual_predictions"],
            "models_used": ensemble_result["models_used"],
            "features_used": 31 if framework == "pytorch" else 26,
            "status": "success",
            "note": "Forced mobile models"
        }

    except Exception as e:
        logger.error(f"❌ Forced mobile prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check with device-specific model info"""
    health_info = {
        "status": "healthy",
        "scaler_loaded": scaler is not None,
        "models_loaded": {}
    }
    
    for device_type in ['desktop', 'mobile']:
        health_info["models_loaded"][device_type] = {
            "pytorch": list(models_info[device_type]['pytorch'].keys()),
            "tensorflow": list(models_info[device_type]['tensorflow'].keys())
        }
    
    return health_info

@app.get("/model-status")
async def model_status():
    """Detailed model status information"""
    status_info = {
        "desktop": {},
        "mobile": {}
    }
    
    for device_type in ['desktop', 'mobile']:
        for framework in ['pytorch', 'tensorflow']:
            status_info[device_type][framework] = {}
            for model_type in ['mlp', 'cnn', 'gru']:
                if model_type in models_info[device_type][framework]:
                    model_info = models_info[device_type][framework][model_type]
                    status_info[device_type][framework][model_type] = {
                        "loaded": True,
                        "type": model_info['type'],
                        "expected_features": model_info['expected_features']
                    }
                else:
                    status_info[device_type][framework][model_type] = {
                        "loaded": False,
                        "type": "not_available"
                    }
    
    return status_info

@app.get("/device-test")
async def device_test(fastapi_request: Request):
    """Test device detection"""
    user_agent = fastapi_request.headers.get('user-agent', '')
    device_type = detect_device_type(user_agent)
    
    return {
        "detected_device": device_type,
        "user_agent": user_agent,
        "available_models": {
            "desktop": {
                "pytorch": list(models_info['desktop']['pytorch'].keys()),
                "tensorflow": list(models_info['desktop']['tensorflow'].keys())
            },
            "mobile": {
                "pytorch": list(models_info['mobile']['pytorch'].keys()),
                "tensorflow": list(models_info['mobile']['tensorflow'].keys())
            }
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Heart Attack Prediction API - Auto Device Detection",
        "endpoints": {
            "POST /predict": "Auto device detection + ensemble predictions",
            "POST /predict/force-desktop": "Force desktop models",
            "POST /predict/force-mobile": "Force mobile models", 
            "GET /health": "System status",
            "GET /model-status": "Detailed model information",
            "GET /device-test": "Test device detection"
        },
        "auto_detection": "Uses User-Agent header to detect mobile vs desktop",
        "fallback_strategy": "If mobile models not found, uses desktop models as fallback"
    }