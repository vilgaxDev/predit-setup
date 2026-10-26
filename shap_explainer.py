"""
shap_explainer.py
=================
Explainable AI engine using SHAP (SHapley Additive exPlanations) for maternal risk models.
Distinguishes statistical associations from clinical causal effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Try to import shap, make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def compute_shap_explanations(model: Any, X_sample: pd.DataFrame) -> Dict[str, Any]:
    """Computes TreeSHAP explanations for tree-based ensemble models."""
    if not SHAP_AVAILABLE:
        return {
            'feature_importance': [],
            'shap_values': None,
            'expected_value': None,
            'error': 'SHAP not available'
        }

    if len(X_sample) == 0:
        return {
            'feature_importance': [],
            'shap_values': None,
            'expected_value': None,
            'error': 'No data provided for SHAP computation'
        }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # Binary classification - use positive class
        if len(shap_values) > 1:
            shap_matrix = shap_values[1]
        else:
            shap_matrix = shap_values[0]
    else:
        # Single class or regression
        shap_matrix = shap_values

    # Handle expected_value format
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        if len(explainer.expected_value) > 1:
            expected_value = float(explainer.expected_value[1])
        else:
            expected_value = float(explainer.expected_value[0])
    else:
        expected_value = float(explainer.expected_value)

    # Mean absolute SHAP across population
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    features = list(X_sample.columns)

    feature_importance = [
        {'feature': feat, 'importance': float(score)}
        for feat, score in sorted(zip(features, mean_abs_shap), key=lambda x: x[1], reverse=True)
    ]

    return {
        'explainer': explainer,
        'shap_matrix': shap_matrix,
        'expected_value': expected_value,
        'feature_importance': feature_importance
    }

def explain_single_patient(
    explainer: Any,
    patient_row: pd.DataFrame,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """Explains an individual patient's prediction using additive Shapley attributions."""
    if not SHAP_AVAILABLE or explainer is None:
        return []

    shap_vals = explainer.shap_values(patient_row)
    if isinstance(shap_vals, list):
        patient_shap = shap_vals[1][0]
    else:
        patient_shap = shap_vals[0]

    explanations = []
    for feat, impact in zip(feature_names, patient_shap):
        val = patient_row[feat].values[0]
        explanations.append({
            'feature': feat,
            'impact': float(impact),
            'value': float(val) if isinstance(val, (int, float, np.number)) else str(val)
        })

    # Sort by absolute impact
    explanations.sort(key=lambda x: abs(x['impact']), reverse=True)
    return explanations
