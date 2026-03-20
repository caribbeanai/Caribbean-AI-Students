"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
 Lesson 04: MLOps & Deployment - Caribbean Crop Disease Detector API
 By Adrian Dunkley
=============================================================================

 Wah gwaan, Graduates! Yuh build di model - now time fi DEPLOY it!

 A model sitting on yuh laptop nah help nobody. Caribbean farmers
 need disease detection in di field, on dem phone, wid spotty
 internet. Dat is di MLOps challenge fi island nations.

 We building a FastAPI-based crop disease detector dat can run
 on edge devices, handle Caribbean connectivity issues, and
 serve farmers from Jamaica to Guyana.

 LEARNING OBJECTIVES:
 1. Build production-ready FastAPI inference endpoint
 2. Model versioning and registry patterns
 3. Edge deployment for low-connectivity environments
 4. Monitoring and observability for ML systems
 5. Caribbean-specific deployment challenges and solutions
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime

# =============================================================================
# PART 1: MODEL REGISTRY - Version Control fi Yuh Models
# =============================================================================

class CaribbeanModelRegistry:
    """
    Model registry fi tracking model versions.
    Like Git but fi trained ML models.

    In production, yuh would use MLflow, Weights & Biases, or DVC.
    """

    def __init__(self):
        self.models: Dict[str, List[Dict]] = {}
        self.active_models: Dict[str, str] = {}

    def register_model(self, name: str, version: str, metrics: Dict,
                       model_path: str, metadata: Dict = None) -> Dict:
        """Register a new model version."""
        if name not in self.models:
            self.models[name] = []

        entry = {
            "name": name,
            "version": version,
            "metrics": metrics,
            "model_path": model_path,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "registered"
        }
        self.models[name].append(entry)
        print(f"Registered: {name} v{version} | accuracy={metrics.get('accuracy', 'N/A')}")
        return entry

    def promote_to_production(self, name: str, version: str):
        """Promote a model version to production."""
        for entry in self.models.get(name, []):
            if entry["version"] == version:
                self.active_models[name] = version
                entry["status"] = "production"
                print(f"Promoted {name} v{version} to PRODUCTION")
                return
        raise ValueError(f"Model {name} v{version} not found")

    def get_production_model(self, name: str) -> Optional[Dict]:
        """Get the current production model."""
        version = self.active_models.get(name)
        if version:
            for entry in self.models[name]:
                if entry["version"] == version:
                    return entry
        return None

    def list_models(self, name: str):
        """List all versions of a model."""
        print(f"\n--- Model: {name} ---")
        for entry in self.models.get(name, []):
            status = entry["status"].upper()
            print(f"  v{entry['version']} | {status} | "
                  f"accuracy={entry['metrics'].get('accuracy', 'N/A')}")


# =============================================================================
# PART 2: CROP DISEASE DETECTOR - Di Model
# =============================================================================

# Caribbean crops and their common diseases
CARIBBEAN_CROP_DISEASES = {
    "banana": {
        "black_sigatoka": {
            "symptoms": "dark streaks on leaves, yellowing, premature ripening",
            "severity": "high",
            "treatment": "fungicide application, resistant varieties, leaf removal",
            "affected_countries": ["Jamaica", "Dominica", "St. Lucia", "St. Vincent"]
        },
        "panama_disease": {
            "symptoms": "yellowing of older leaves, wilting, pseudostem splitting",
            "severity": "critical",
            "treatment": "no chemical cure - use resistant cultivars, quarantine",
            "affected_countries": ["Jamaica", "Trinidad", "Guyana", "Suriname"]
        },
    },
    "sugarcane": {
        "smut": {
            "symptoms": "black whip-like structure from growing point",
            "severity": "high",
            "treatment": "resistant varieties, hot water treatment of setts",
            "affected_countries": ["Guyana", "Barbados", "Jamaica", "Trinidad"]
        },
        "rust": {
            "symptoms": "orange-brown pustules on leaves",
            "severity": "medium",
            "treatment": "resistant varieties, fungicide in severe cases",
            "affected_countries": ["Barbados", "Guyana", "Jamaica"]
        },
    },
    "cocoa": {
        "black_pod": {
            "symptoms": "dark brown to black lesions on pods",
            "severity": "high",
            "treatment": "frequent harvesting, fungicide, shade management",
            "affected_countries": ["Trinidad", "Grenada", "Jamaica", "Dominica"]
        },
        "frosty_pod": {
            "symptoms": "white powdery coating on pods, internal rot",
            "severity": "critical",
            "treatment": "removal of infected pods, biological control",
            "affected_countries": ["Trinidad", "Grenada"]
        },
    },
    "citrus": {
        "citrus_greening": {
            "symptoms": "blotchy mottled leaves, lopsided bitter fruit",
            "severity": "critical",
            "treatment": "no cure - remove infected trees, control psyllid vector",
            "affected_countries": ["Jamaica", "Belize", "Cuba", "Trinidad"]
        },
    },
    "coconut": {
        "lethal_yellowing": {
            "symptoms": "premature nut drop, yellowing fronds, crown collapse",
            "severity": "critical",
            "treatment": "antibiotic injection (temporary), resistant varieties",
            "affected_countries": ["Jamaica", "Haiti", "Cuba", "Bahamas"]
        },
    },
}


class CropDiseaseModel:
    """
    Simulated crop disease detection model.
    In production, dis would be a CNN (ResNet/EfficientNet) trained
    on images of diseased crops.
    """

    def __init__(self, model_name: str = "caribbean-crop-disease-v1"):
        self.model_name = model_name
        self.classes = []
        self._build_classes()

    def _build_classes(self):
        """Build class list from disease database."""
        self.classes = ["healthy"]
        for crop, diseases in CARIBBEAN_CROP_DISEASES.items():
            for disease in diseases:
                self.classes.append(f"{crop}_{disease}")
        print(f"Model initialized with {len(self.classes)} classes")

    def predict(self, image_features: np.ndarray) -> Dict:
        """
        Predict crop disease from image features.
        Returns class probabilities and top prediction.
        """
        np.random.seed(hash(str(image_features.sum())) % 2**31)
        logits = np.random.randn(len(self.classes))
        logits[0] += 0.5  # Slight bias toward healthy

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        top_idx = np.argmax(probs)
        return {
            "predicted_class": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "model_version": self.model_name
        }


# =============================================================================
# PART 3: FASTAPI APPLICATION (Structure / Pseudocode)
# =============================================================================

FASTAPI_APP_CODE = '''
"""
Caribbean Crop Disease Detector API
Deploy wid: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
import time
import logging

app = FastAPI(
    title="Caribbean Crop Disease Detector",
    description="AI-powered crop disease detection for Caribbean farmers",
    version="1.0.0"
)

# CORS - allow mobile apps from across di islands to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Response Models ---
class PredictionResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    severity: str
    treatment: str
    is_healthy: bool
    inference_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str

# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check fi load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        uptime_seconds=time.time() - app.state.start_time,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a crop image and get disease prediction.
    Supports: JPEG, PNG
    Max size: 10MB (optimized fi mobile upload on Caribbean networks)
    """
    start = time.time()

    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    # Read and preprocess image
    contents = await file.read()
    # image = preprocess_image(contents)  # Resize, normalize

    # Run inference
    # prediction = model.predict(image)

    inference_time = (time.time() - start) * 1000

    return PredictionResponse(
        crop="banana",
        disease="black_sigatoka",
        confidence=0.89,
        severity="high",
        treatment="Apply fungicide, remove affected leaves",
        is_healthy=False,
        inference_time_ms=inference_time,
        model_version="v1.0.0"
    )

@app.get("/diseases/{crop}")
async def get_crop_diseases(crop: str):
    """Get all known diseases for a specific Caribbean crop."""
    diseases = CARIBBEAN_CROP_DISEASES.get(crop.lower())
    if not diseases:
        raise HTTPException(404, f"Crop '{crop}' not found")
    return {"crop": crop, "diseases": diseases}

@app.get("/crops")
async def list_supported_crops():
    """List all supported crops."""
    return {"crops": list(CARIBBEAN_CROP_DISEASES.keys())}
'''


# =============================================================================
# PART 4: MONITORING & OBSERVABILITY
# =============================================================================

class ModelMonitor:
    """
    Monitor model performance in production.
    Catch data drift, model degradation, and system issues.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.predictions_log: List[Dict] = []
        self.alert_thresholds = {
            "min_confidence": 0.3,
            "max_latency_ms": 500,
            "drift_threshold": 0.1,
        }

    def log_prediction(self, prediction: Dict, latency_ms: float):
        """Log a prediction for monitoring."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction["predicted_class"],
            "confidence": prediction["confidence"],
            "latency_ms": latency_ms,
        }
        self.predictions_log.append(entry)
        self._check_alerts(entry)

    def _check_alerts(self, entry: Dict):
        """Check if any monitoring thresholds are breached."""
        if entry["confidence"] < self.alert_thresholds["min_confidence"]:
            print(f"  ALERT: Low confidence prediction ({entry['confidence']:.2f})")
        if entry["latency_ms"] > self.alert_thresholds["max_latency_ms"]:
            print(f"  ALERT: High latency ({entry['latency_ms']:.0f}ms)")

    def compute_metrics(self) -> Dict:
        """Compute monitoring metrics over recent predictions."""
        if not self.predictions_log:
            return {}

        confidences = [p["confidence"] for p in self.predictions_log]
        latencies = [p["latency_ms"] for p in self.predictions_log]

        return {
            "total_predictions": len(self.predictions_log),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "low_confidence_rate": np.mean([c < 0.5 for c in confidences]),
        }

    def detect_data_drift(self, reference_dist: np.ndarray,
                          current_dist: np.ndarray) -> Dict:
        """
        Detect if incoming data distribution has shifted.
        Uses simple KL divergence approximation.
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ref = reference_dist + eps
        cur = current_dist + eps

        # Normalize
        ref = ref / ref.sum()
        cur = cur / cur.sum()

        # KL divergence
        kl_div = np.sum(ref * np.log(ref / cur))

        drifted = kl_div > self.alert_thresholds["drift_threshold"]
        return {
            "kl_divergence": float(kl_div),
            "drift_detected": drifted,
            "threshold": self.alert_thresholds["drift_threshold"]
        }


# =============================================================================
# PART 5: EDGE DEPLOYMENT - Fi Caribbean Farmers in di Field
# =============================================================================

class EdgeDeploymentConfig:
    """
    Configuration fi deploying models on edge devices.
    Caribbean reality: intermittent connectivity, older phones,
    limited bandwidth, high temperatures.
    """

    def __init__(self):
        self.configs = {
            "mobile_phone": {
                "model_format": "TFLite / ONNX",
                "max_model_size_mb": 50,
                "quantization": "INT8",
                "target_latency_ms": 200,
                "offline_capable": True,
                "notes": "Most farmers have Android phones. Model must work offline "
                         "fi when dem in di bush wid no signal."
            },
            "raspberry_pi": {
                "model_format": "TFLite / ONNX",
                "max_model_size_mb": 100,
                "quantization": "INT8 or FP16",
                "target_latency_ms": 500,
                "offline_capable": True,
                "notes": "Solar-powered Pi at farm entrance. Farmer bring leaf, "
                         "camera snap, instant diagnosis. No internet needed."
            },
            "cloud_api": {
                "model_format": "PyTorch / TensorFlow",
                "max_model_size_mb": 2000,
                "quantization": "FP32",
                "target_latency_ms": 100,
                "offline_capable": False,
                "notes": "Full model in cloud. Best accuracy but need internet. "
                         "Host in Miami or Sao Paulo fi low latency to Caribbean."
            },
            "hybrid": {
                "model_format": "TFLite (edge) + PyTorch (cloud)",
                "max_model_size_mb": 50,
                "quantization": "INT8 (edge) + FP32 (cloud)",
                "target_latency_ms": 200,
                "offline_capable": True,
                "notes": "Small model on phone fi quick triage. When internet "
                         "available, send to cloud fi detailed analysis. "
                         "Best of both worlds fi Caribbean context."
            }
        }

    def recommend(self, connectivity: str, device: str) -> Dict:
        """Recommend deployment strategy based on Caribbean constraints."""
        if connectivity == "none":
            return self.configs["mobile_phone" if device == "phone" else "raspberry_pi"]
        elif connectivity == "intermittent":
            return self.configs["hybrid"]
        else:
            return self.configs["cloud_api"]

    def show_all_configs(self):
        """Display all deployment configurations."""
        print("\n" + "=" * 60)
        print(" EDGE DEPLOYMENT CONFIGURATIONS")
        print("=" * 60)
        for name, config in self.configs.items():
            print(f"\n--- {name.upper()} ---")
            for key, val in config.items():
                print(f"  {key}: {val}")


# =============================================================================
# PART 6: QUIZ
# =============================================================================

QUIZ_QUESTIONS = """
=============================================================================
 QUIZ: MLOps & DEPLOYMENT (10 Questions)
=============================================================================

Q1: Why is edge deployment especially important for Caribbean AI applications?
    a) It is cheaper
    b) Many rural areas have intermittent or no internet connectivity,
       so models must run locally on device
    c) Edge devices are faster
    d) Cloud computing is not available in the Caribbean

Q2: What is model quantization (INT8) and why use it for mobile deployment?
    a) It makes the model more accurate
    b) It reduces model size and inference time by using 8-bit integers
       instead of 32-bit floats, with minimal accuracy loss
    c) It encrypts the model
    d) It increases the model's vocabulary

Q3: What does a /health endpoint in an ML API typically check?
    a) The user's health data
    b) Whether the service is running, the model is loaded,
       and the system can accept requests
    c) The training data quality
    d) Network connectivity

Q4: What is data drift in production ML systems?
    a) When the database crashes
    b) When the distribution of incoming data changes compared to
       training data, potentially degrading model performance
    c) When the model weights change
    d) When users stop using the system

Q5: For a Caribbean crop disease app, why is the hybrid deployment
    approach (edge + cloud) recommended?
    a) It is the cheapest option
    b) Quick local inference when offline, with cloud refinement when
       connected, matching the intermittent connectivity reality
    c) It uses the least storage
    d) It is easier to build

Q6: What is the purpose of a model registry?
    a) To store training data
    b) To track model versions, metrics, and manage promotion from
       staging to production
    c) To register users
    d) To log predictions

Q7: What monitoring metric would alert you to a potential model
    degradation in production?
    a) Number of API calls
    b) Sustained drop in average prediction confidence or increase
       in low-confidence predictions
    c) Server CPU temperature
    d) Number of registered users

Q8: Why should Caribbean ML APIs be hosted in nearby cloud regions
    (e.g., Miami, Sao Paulo) rather than distant ones?
    a) It is cheaper
    b) Lower network latency means faster response times for users
       across the Caribbean islands
    c) Better security
    d) More storage available

Q9: What file format is commonly used for mobile ML deployment?
    a) .h5 (Keras)
    b) .pt (PyTorch)
    c) .tflite (TensorFlow Lite) or .onnx (ONNX)
    d) .csv

Q10: In the context of Caribbean agriculture AI, what is the biggest
     challenge for building training datasets?
    a) Too much data available
    b) Limited labeled images of Caribbean-specific crop diseases,
       requiring partnerships with local agricultural extension services
    c) Images are too high quality
    d) Crops grow too fast to photograph
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - GRADUATES MODULE")
    print(" Lesson 04: MLOps & Deployment")
    print(" Caribbean Crop Disease Detector API")
    print(" By Adrian Dunkley")
    print("=" * 60)

    # Model Registry demo
    registry = CaribbeanModelRegistry()
    registry.register_model("crop-disease", "1.0", {"accuracy": 0.82}, "models/v1.0.pt")
    registry.register_model("crop-disease", "1.1", {"accuracy": 0.87}, "models/v1.1.pt")
    registry.register_model("crop-disease", "2.0", {"accuracy": 0.91}, "models/v2.0.pt")
    registry.promote_to_production("crop-disease", "2.0")
    registry.list_models("crop-disease")

    # Disease model demo
    model = CropDiseaseModel()
    test_features = np.random.randn(224)
    prediction = model.predict(test_features)
    print(f"\nSample prediction: {prediction['predicted_class']} "
          f"({prediction['confidence']:.2%})")

    # Monitoring demo
    monitor = ModelMonitor("crop-disease-v2.0")
    for _ in range(20):
        pred = model.predict(np.random.randn(224))
        latency = np.random.exponential(50) + 20
        monitor.log_prediction(pred, latency)

    metrics = monitor.compute_metrics()
    print(f"\nMonitoring Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Edge deployment configs
    edge = EdgeDeploymentConfig()
    edge.show_all_configs()

    # Show FastAPI code
    print("\n" + "=" * 60)
    print(" FASTAPI APPLICATION CODE")
    print("=" * 60)
    print(FASTAPI_APP_CODE)

    # Quiz
    print(QUIZ_QUESTIONS)

    print("\n" + "=" * 60)
    print(" Now yuh can deploy models fi real Caribbean impact!")
    print(" Next: RAG Systems - intelligent knowledge retrieval!")
    print("=" * 60)
