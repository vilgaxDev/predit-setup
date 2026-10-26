"""
test_pipeline.py
================
Automated test suite for data validation, de-identification, model training,
and small-cell privacy suppression.
"""

import pytest
import pandas as pd
import numpy as np

from data_pipeline import sanitize_and_deidentify, apply_small_cell_suppression
from model_engine import engineer_features, train_and_evaluate_model
from pmtct_tracker import calculate_seroconversion_metrics, calculate_pmtct_five_year_cascade

def test_deidentification_strips_pii():
    test_df = pd.DataFrame({
        'mother_pseudonym': ['KEN-MOM-101'],
        'patient_name': ['Jane Doe'],
        'phone_number': ['+254712345678'],
        'national_id': ['12345678'],
        'systolic_bp': [120]
    })
    cleaned = sanitize_and_deidentify(test_df)
    assert 'patient_name' not in cleaned.columns
    assert 'phone_number' not in cleaned.columns
    assert 'national_id' not in cleaned.columns
    assert 'mother_pseudonym' in cleaned.columns
    assert 'systolic_bp' in cleaned.columns

def test_small_cell_suppression():
    counts = pd.Series([0, 2, 4, 5, 12, np.nan])
    suppressed = apply_small_cell_suppression(counts, threshold=5)
    assert suppressed.iloc[1] == "<5"
    assert suppressed.iloc[2] == "<5"
    assert suppressed.iloc[3] == "5.0" or suppressed.iloc[3] == "5"
    assert suppressed.iloc[4] == "12.0" or suppressed.iloc[4] == "12"

def test_seroconversion_calculation():
    test_data = pd.DataFrame({
        'mother_pseudonym': ['M1', 'M2', 'M3', 'M4'],
        'county': ['Nairobi', 'Nairobi', 'Kisumu', 'Kisumu'],
        'hiv_status_at_enrolment': ['Negative', 'Negative', 'Negative', 'Positive'],
        'seroconversion_status': ['Confirmed Seroconversion', 'Negative Maintained', 'Negative Maintained', 'Known Positive at Baseline']
    })
    metrics = calculate_seroconversion_metrics(test_data)
    assert metrics['total_eligible_baseline_negative'] == 3
    assert metrics['confirmed_seroconversions'] == 1
    assert metrics['incidence_per_1000_pregnant_women'] > 0

def test_model_training_pipeline():
    # Synthetic mini dataset
    data = pd.DataFrame({
        'maternal_age': [24, 38, 16, 29, 32],
        'gravida': [2, 4, 1, 3, 2],
        'parity': [1, 3, 0, 2, 1],
        'gestational_age_weeks': [32, 36, 28, 30, 38],
        'total_anc_visits': [3, 4, 1, 5, 4],
        'systolic_bp': [120, 160, 110, 142, 118],
        'diastolic_bp': [80, 105, 70, 92, 78],
        'haemoglobin': [12.0, 9.2, 11.5, 6.8, 12.4],
        'blood_glucose': [5.0, 6.2, 4.9, 8.5, 5.1],
        'proteinuria': ['Negative', '2+', 'Negative', '1+', 'Negative'],
        'hiv_status': ['Negative', 'Positive', 'Negative', 'Negative', 'Negative'],
        'arv_adherence': ['Not Applicable', 'Good', 'Not Applicable', 'Not Applicable', 'Not Applicable'],
        'diabetes_status': ['Normal', 'Normal', 'Normal', 'Gestational DM', 'Normal'],
        'maternal_complications': [['None'], ['Pre-eclampsia'], ['None'], ['Postpartum Haemorrhage (PPH)'], ['None']]
    })
    
    X, y = engineer_features(data)
    assert len(X) == 5
    assert len(y) == 5
    assert 'systolic_bp' in X.columns
