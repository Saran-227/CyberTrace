"""Location classification model interface for predicting cash-out withdrawal zones."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.utils.logging import get_logger

logger = get_logger("LocationClassifier")

MODEL_FACTORIES = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
}

class LocationClassifier:
    """End-to-end wrapper uniting candidate preprocessor pipeline with classification estimator."""

    def __init__(self, model_name: str = "Random Forest"):
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Unsupported model family: '{model_name}'. Choose from: {list(MODEL_FACTORIES.keys())}")
        self.model_name = model_name
        self.preprocessor = build_preprocessing_pipeline()
        self.estimator = MODEL_FACTORIES[model_name]()
        self.pipeline: Optional[Pipeline] = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", self.estimator),
        ])
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LocationClassifier":
        """Fit preprocessing pipeline and classifier on training data."""
        logger.info(f"Training {self.model_name} on {len(X)} records...")
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_
        self.is_fitted = True
        logger.info(f"Model training complete. Target classes: {list(self.classes_)}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target withdrawal zones."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute class probability distribution across withdrawal zones."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict_proba(X)

    def save(self, filepath: Path) -> None:
        """Serialize trained pipeline to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model pipeline.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model pipeline saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "LocationClassifier":
        """Load serialized pipeline from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        pipeline = joblib.load(filepath)
        classifier_step = pipeline.named_steps["classifier"]
        model_name = type(classifier_step).__name__

        instance = cls.__new__(cls)
        instance.model_name = model_name
        instance.pipeline = pipeline
        instance.preprocessor = pipeline.named_steps["preprocessor"]
        instance.estimator = classifier_step
        instance.classes_ = classifier_step.classes_
        instance.is_fitted = True
        logger.info(f"Model pipeline loaded successfully from: {filepath}")
        return instance
