import logging
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_model(model, val_loader) -> Dict[str, float]:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in val_loader:
            preds = model(features.to(next(model.parameters()).device)).cpu().numpy()
            predictions.append(preds.reshape(preds.shape[0], -1))
            targets.append(labels.cpu().numpy().reshape(labels.shape[0], -1))

    if not predictions:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)

    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
