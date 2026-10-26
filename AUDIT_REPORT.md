# AUDIT REPORT: Health Data Anonymisation and Predictive Monitoring Project

**Date:** 2026-10-14
**Auditor:** Devin AI Agent
**Scope:** Entire Python/Streamlit project for maternal-child health data anonymisation and predictive modeling

---

## EXECUTIVE SUMMARY

**CRITICAL FINDINGS:**
1. **NER evaluation is entirely simulated** - No ground-truth annotations exist; metrics are calculated from synthetic predictions with injected noise
2. **Predictive model uses synthetic targets** - No genuine clinical outcome labels; targets are algorithmically generated from data distributions
3. **Misleading dashboard labels** - Dashboard claims "ground-truth evaluation completed" and "genuine outcome labels" when neither exists
4. **No leakage-resistant evaluation** - No patient-level grouping, no temporal split, no proper train/validation/test isolation
5. **No reproducibility safeguards** - No dataset fingerprinting, no model versioning, no automated tests

**RECOMMENDATION:** Do not use current metrics for any publication, clinical decision-making, or claims of model performance. The system requires fundamental redesign before deployment.

---

## 1. DATA AUDIT

### 1.1 Dataset Verification
**Claim:** 12,000 anonymised records with 44 clinical features
**Verification:** ✅ CONFIRMED
- Actual records: 12,000
- Actual columns: 44
- Source: `anonymized_data_kg_export.xlsx`

### 1.2 Data Schema Analysis
**Columns Found:**
```
['Date_of_Admission', 'Admission_Number', 'No. of ANC Visits', 'Name', 'Village', 
 'Age', 'Marital Status', 'Parity', 'Gravidae', 'Date of LMP', 'EDD', 
 'Gestation at Birth (Weeks)', 'Diagnosis', 'Duration at onset of labour', 
 'Time of Delivery', 'Date of Delivery', 'Mode_of_Delivery', 'Placenta Complete', 
 'Blood loss (in mls)', "Mother's Status after Delivery", 
 'Maternal deaths audited (date)', 'Delivery Complications (codes)', 
 'Sex(m/f)', 'Birth Weight (in grams)', 'Live Birth / FSB / MSB', 
 'Initiation of BF within 1 HR (Y/N)', 'APGAR Score', 
 'Tetracycline at birth(Y/N)', 'Birth with deformities(Y/N)', 
 'VDRL/ RPR Results(P/N)', 'HIV Status (P/N/KP/U)', 'Regimen to Mother (Y/N)', 
 'Regimen To Baby(Codes)', 'CTX to Mother(Y/N)', 'Tested\t(Y/N)', 
 'HIV Test Results\t(Y/N)', 'Vitamin A supplementation (Y/N)', 
 'Counselled on feeding options\t(1,2)', 'Delivery Conducted by', 
 'Birth Notification Number', 'Discharge date(dd/mm/yy)', 
 'Status of the baby(A/D)', 'Referrals', 'Comments']
```

**Issues:**
- ❌ No unique patient identifier to detect duplicate records
- ❌ No clear target variable for prediction
- ❌ Date fields partially anonymised (day hidden in `Date_of_Admission`)
- ❌ No temporal information for time-series evaluation
- ❌ Mixed data types (strings, ints, floats) with no type validation

### 1.3 Patient-Level Verification
**Status:** ❌ NOT VERIFIED
- Cannot determine unique patient count
- Cannot detect duplicate encounters
- Cannot verify repeated records per mother
- **Impact:** Risk of data leakage if same patient appears in train and test sets

---

## 2. NER EVALUATION AUDIT

### 2.1 Current Implementation
**File:** `run_notebook.py` (lines 319-434)

**Critical Issue: SIMULATED EVALUATION**
```python
# Line 322-324
print("Note: Simulating realistic NER model performance")
print("      (introducing 5-10% error rate typical of real NER models)")

# Line 329-357
def simulate_ner_predictions(all_bio_labels, error_rate=0.08):
    """Simulate realistic NER predictions by introducing errors"""
    # Randomly flips labels to simulate model errors
```

**What Actually Happens:**
1. System generates BIO tags from rule-based entity map
2. System injects 8% random errors into these tags
3. System compares "simulated predictions" against "original rule-based tags"
4. Metrics calculated from this comparison
5. **NO actual NER model is trained or evaluated**

### 2.2 Dashboard Claims vs Reality
**Dashboard Text (app.py line 1158):**
```
"Comparison of predicted PII labels against independently verified annotations"
```

**Dashboard Text (app.py line 1163):**
```
"Error Rate: 8% injected label noise for realistic simulation"
```

**Dashboard Text (app.py line 1185):**
```
"✅ Ground-truth evaluation completed with verified annotations"
```

**Reality:**
- ❌ No independently verified annotations exist
- ❌ No ground-truth labels available
- ❌ Metrics are from simulated predictions, not real model
- ❌ Claim of "verified annotations" is FALSE

### 2.3 Missing Components
**Required for Ground-Truth Evaluation:**
- ❌ Manual annotation of PII in clinical records
- ❌ Inter-annotator agreement metrics
- ❌ Annotation guidelines
- ❌ Separate development and evaluation sets
- ❌ Actual trained NER model (BERT fine-tuned)
- ❌ Entity-level precision/recall on real annotations
- ❌ Error analysis (missed PII, false detections, incorrect types)

### 2.4 Recommended Fix
**Option A (If annotations available):**
1. Load verified ground-truth annotations
2. Run actual NER model predictions
3. Calculate entity-level metrics
4. Display "Evaluation on real held-out data"

**Option B (If annotations unavailable):**
1. Display "Ground truth unavailable — evaluation pending"
2. Provide annotation workflow interface
3. Do NOT claim verified evaluation
4. Do NOT display target attainment metrics

---

## 3. PREDICTIVE MODEL AUDIT

### 3.1 Target Variable Issue
**File:** `model_engine.py` (lines 36-50)

**Critical Issue: SYNTHETIC TARGET**
```python
# Line 38-42
if 'maternal_complications' in df.columns:
    y = df['maternal_complications'].apply(
        lambda x: 1 if isinstance(x, str) and ('complication' in str(x).lower() or 'pph' in str(x).lower()) else 0
    )
elif 'Mode_of_Delivery' in df.columns:
    y = df['Mode_of_Delivery'].apply(lambda x: 1 if 'CS' in str(x) else 0)
else:
    # Line 46-48
    y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
```

**Problems:**
- ❌ No genuine clinical outcome labels
- ❌ Target is algorithmically generated from data distribution
- ❌ No clinical definition of outcome
- ❌ No observation window specified
- ❌ No documentation of when predictors would be available
- ❌ Target changes based on which columns exist (not reproducible)

### 3.2 Feature Engineering Issues
**File:** `model_engine.py` (lines 22-52)

**Issues:**
- ❌ Uses ALL numeric columns without clinical validation
- ❌ No exclusion of identifiers
- ❌ No exclusion of post-outcome variables
- ❌ No exclusion of target-derived features
- ❌ No feature selection based on clinical relevance
- ❌ No handling of missing data beyond fillna(0)

### 3.3 Data Split Issues
**File:** `model_engine.py` (lines 84-87)

**Critical Issue: NO LEAKAGE PREVENTION**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, 
    stratify=y if y.nunique() > 1 else None
)
```

**Problems:**
- ❌ No patient-level grouping (same mother in train and test)
- ❌ No temporal split (evaluating on future data)
- ❌ No separate validation set for model selection
- ❌ No stratification by clinical subgroups
- ❌ No consideration of repeated encounters

### 3.4 Model Training Issues
**File:** `model_engine.py` (lines 89-96)

**Issues:**
- ❌ No baseline models (logistic regression, decision tree)
- ❌ Hyperparameters hardcoded (n_estimators=120, learning_rate=0.08)
- ❌ No hyperparameter tuning on validation set
- ❌ No cross-validation
- ❌ No model selection process

### 3.5 Metric Calculation Issues
**File:** `model_engine.py` (lines 98-108)

**Issues:**
- ❌ No confidence intervals
- ❌ No per-class metrics
- ❌ No macro/weighted averages
- ❌ No precision-recall AUC
- ❌ No calibration metrics (Brier score, calibration slope/intercept)
- ❌ No threshold analysis
- ❌ No subgroup evaluation

### 3.6 Dashboard Claims vs Reality
**Dashboard Text (app.py line 1191):**
```
"Train, test and calculate metrics using genuine outcome labels from clinical records"
```

**Dashboard Text (app.py line 1200):**
```
"Target: Adverse clinical outcomes (based on genuine labels)"
```

**Reality:**
- ❌ No genuine outcome labels
- ❌ Target is synthetic/algorithmic
- ❌ No clinical definition of outcome
- ❌ Claims are FALSE

### 3.7 Reported Metrics Analysis
**Reported Metrics:**
- Accuracy: 91.07%
- Precision: 86.92%
- Recall: 63.70%
- F1: 73.52%
- ROC AUC: 87.33%
- Mean predicted probability: 0.501
- Mean actual rate: 0.450

**Issues:**
- ❌ Metrics calculated on synthetic target
- ❌ No verification against actual predictions
- ❌ No confidence intervals
- ❌ No comparison to baseline
- ❌ Calibration shows mismatch (0.501 vs 0.450) but no interpretation

---

## 4. SHAP EXPLANATION AUDIT

### 4.1 Implementation Review
**File:** `shap_explainer.py`

**Assessment:** ✅ Implementation is technically correct
- Uses TreeExplainer for gradient boosting
- Computes mean absolute SHAP values
- Handles binary classification correctly

### 4.2 Dependency Issues
**Critical Dependency on Model Quality:**
- ❌ SHAP explanations are only meaningful if model is valid
- ❌ Since model uses synthetic targets, explanations are misleading
- ❌ Feature names may not align with clinical meaning
- ❌ No verification that SHAP values align with clinical knowledge

### 4.3 Dashboard Claims
**Dashboard Text (app.py line 856):**
```
"Feature attribution via Shapley Additive Explanations. 
Distinguishes statistical associations from causal clinical pathways."
```

**Reality:**
- ⚠️ SHAP shows statistical associations, NOT causal pathways
- ⚠️ Cannot distinguish causality from correlation
- ⚠️ No clinical validation of feature importance
- ⚠️ Explanations may be misleading given synthetic target

---

## 5. DASHBOARD AUDIT

### 5.1 Misleading Success Indicators
**Locations:**
- app.py line 577: "Target met" / "Below target" for simulated metrics
- app.py line 1185: "✅ Ground-truth evaluation completed with verified annotations"
- app.py line 488: "✅ ML model trained successfully"
- app.py line 505: "✅ Model saved for SHAP explanations"

**Issues:**
- ❌ Green checkmarks imply validation when none exists
- ❌ No evaluation status indicator (simulated vs real)
- ❌ No distinction between simulated and genuine results
- ❌ Misleading to users and reviewers

### 5.2 Missing Audit-Ready Information
**Required for Auditability:**
- ❌ Dataset version/fingerprint
- ❌ Record counts and unique-patient counts
- ❌ Split sizes (train/validation/test)
- ❌ Annotation counts and evaluation protocol
- ❌ Model version and timestamp
- ❌ Target definition and observation window
- ❌ Feature list with clinical definitions
- ❌ Downloadable predictions with metadata
- ❌ Reproducibility metadata (random seeds, library versions)

### 5.3 Auto-Run Concerns
**File:** app.py (lines 217-280)

**Issue:**
- ❌ Auto-runs pipeline on first load
- ❌ Auto-trains model without user consent
- ❌ No way to distinguish auto-run from manual run
- ❌ No audit trail of what was executed

---

## 6. LEAKAGE RISK ANALYSIS

### 6.1 Identified Leakage Sources
1. **Patient-Level Leakage:**
   - No grouping by patient ID
   - Same mother could be in train and test
   - repeated encounters not handled

2. **Temporal Leakage:**
   - No time-based split
   - May train on future data to predict past
   - No observation window enforcement

3. **Feature Leakage:**
   - All numeric columns used without filtering
   - Potential post-outcome variables included
   - No exclusion of identifiers

4. **Target Leakage:**
   - Target derived from same data as features
   - No clinical definition of temporal precedence
   - No verification of predictor availability at prediction time

### 6.2 Leakage Prevention Required
**Missing:**
- ❌ Patient-level grouping in train/test split
- ❌ Temporal split (train on earlier data, test on later)
- ❌ Clinical definition of when predictors available
- ❌ Exclusion of post-outcome variables
- ❌ Exclusion of identifiers
- ❌ Feature leakage audit

---

## 7. REPRODUCIBILITY AUDIT

### 7.1 Missing Safeguards
**Required:**
- ❌ Random seed documentation
- ❌ Library version pinning (requirements.txt incomplete)
- ❌ Dataset checksum/fingerprint
- ❌ Model serialization with version info
- ❌ Deterministic preprocessing
- ❌ Automated tests

### 7.2 Current State
**File:** `requirements.txt`
```
streamlit
pandas
numpy
openpyxl
transformers
matplotlib
seaborn
networkx
scikit-learn
plotly
shap
```

**Issues:**
- ❌ No version pinning (all versions can float)
- ❌ No dependency resolution
- ❌ No environment specification
- ❌ Not reproducible across installations

---

## 8. AUTOMATED TESTS AUDIT

### 8.1 Current State
**Status:** ❌ NO AUTOMATED TESTS EXIST

**Required Tests:**
- ❌ Metric calculation tests (compare against known examples)
- ❌ Patient-level split isolation tests
- ❌ Target leakage detection tests
- ❌ Reproducibility tests (same inputs → same outputs)
- ❌ Model loading/saving tests
- ❌ SHAP output consistency tests
- ❌ Dashboard-metric agreement tests

---

## 9. SUMMARY OF CRITICAL ISSUES

### 9.1 High Priority (Must Fix Before Any Use)
1. **NER evaluation is simulated** - No ground truth exists
2. **Predictive target is synthetic** - No genuine clinical outcomes
3. **Dashboard claims are misleading** - False claims of validation
4. **No leakage prevention** - Patient-level and temporal leakage likely
5. **No proper train/validation/test split** - All data in single split

### 9.2 Medium Priority (Must Fix for Publication)
1. **No confidence intervals** - Metrics without uncertainty quantification
2. **No baseline models** - No comparison to simple models
3. **No subgroup evaluation** - No performance by clinical subgroups
4. **No reproducibility safeguards** - No version pinning, no tests
5. **No audit trail** - No dataset fingerprinting, no model versioning

### 9.3 Low Priority (Should Fix for Best Practices)
1. **No error analysis** - No breakdown of model errors
2. **No threshold analysis** - No sensitivity/FP trade-off analysis
3. **No calibration metrics** - No Brier score, calibration slope
4. **No annotation workflow** - No interface for ground-truth creation
5. **No automated tests** - No regression testing

---

## 10. RECOMMENDED ACTION PLAN

### Phase 1: Immediate Actions (Critical)
1. **Add evaluation status indicator** to dashboard
   - Label all simulated metrics as "SIMULATED"
   - Label all synthetic targets as "SYNTHETIC"
   - Remove misleading green checkmarks
   - Add "Ground truth unavailable" banner

2. **Document data limitations**
   - Add dataset fingerprint section
   - Document synthetic target generation
   - Document missing ground-truth annotations
   - Document leakage risks

3. **Add disclaimer**
   - "Not for clinical use"
   - "Metrics are simulated/synthetic"
   - "Requires ground-truth annotations for validation"
   - "Not externally validated"

### Phase 2: Data Preparation (Required for Real Evaluation)
1. **Create patient identifier**
   - Add unique patient ID
   - Identify duplicate records
   - Count unique patients vs records

2. **Define clinical outcome**
   - Document clinical definition
   - Specify observation window
   - Document predictor availability
   - Exclude post-outcome variables

3. **Create train/validation/test split**
   - Patient-level grouping
   - Temporal split if possible
   - Stratify by key variables
   - Document split sizes

### Phase 3: Ground-Truth NER Evaluation (Required)
1. **Manual annotation**
   - Annotate PII in sample records
   - Create annotation guidelines
   - Calculate inter-annotator agreement
   - Create separate dev/eval sets

2. **Train actual NER model**
   - Fine-tune BERT on annotated data
   - Evaluate on held-out test set
   - Calculate entity-level metrics
   - Perform error analysis

### Phase 4: Proper Predictive Model (Required)
1. **Use genuine outcome labels**
   - Use actual clinical outcomes
   - Document clinical definition
   - Verify temporal precedence
   - Exclude leakage sources

2. **Implement proper evaluation**
   - Train/validation/test split
   - Baseline models
   - Hyperparameter tuning
   - Confidence intervals
   - Subgroup evaluation

3. **Add model comparison**
   - Compare to simple baselines
   - Report performance with uncertainty
   - Document limitations
   - Avoid overclaiming

### Phase 5: Reproducibility (Best Practice)
1. **Version pinning**
   - Pin all library versions
   - Document Python version
   - Create environment.yml
   - Add checksum validation

2. **Automated tests**
   - Metric calculation tests
   - Split isolation tests
   - Leakage detection tests
   - Reproducibility tests

3. **Audit trail**
   - Dataset fingerprinting
   - Model versioning
   - Execution logging
   - Metadata export

---

## 11. CONCLUSION

**Current State:** The project is NOT scientifically rigorous, reproducible, or independently auditable in its current form.

**Critical Blockers:**
1. No ground-truth NER annotations
2. No genuine clinical outcome labels
3. No leakage-resistant evaluation
4. Misleading dashboard claims

**Path Forward:**
1. **Immediate:** Add disclaimers and evaluation status indicators
2. **Short-term:** Create ground-truth annotations and genuine outcomes
3. **Medium-term:** Implement proper evaluation with leakage prevention
4. **Long-term:** Add reproducibility safeguards and automated tests

**Warning:** Do not use current metrics for any publication, clinical decision-making, or claims of model performance. The system requires fundamental redesign before deployment.

---

## APPENDIX: FILES REQUIRING MODIFICATION

1. **app.py** - Dashboard UI
   - Add evaluation status indicators
   - Remove misleading success messages
   - Add dataset fingerprint display
   - Add disclaimer banners

2. **run_notebook.py** - Pipeline script
   - Remove simulated NER evaluation
   - Add ground-truth annotation loading
   - Remove synthetic target generation
   - Add patient-level split

3. **model_engine.py** - ML engine
   - Remove synthetic target generation
   - Add proper feature engineering
   - Add baseline models
   - Add confidence intervals
   - Add subgroup evaluation

4. **shap_explainer.py** - SHAP engine
   - Add clinical interpretation warnings
   - Add feature name validation
   - Add reproducibility checks

5. **requirements.txt** - Dependencies
   - Pin all versions
   - Add Python version
   - Add development dependencies

6. **NEW: evaluation.py** - Evaluation module
   - Ground-truth NER evaluation
   - Proper predictive evaluation
   - Leakage detection
   - Confidence intervals

7. **NEW: tests/** - Test suite
   - Metric calculation tests
   - Split isolation tests
   - Reproducibility tests

8. **NEW: annotation_workflow.py** - Annotation interface
   - Manual PII annotation
   - Inter-annotator agreement
   - Ground-truth export

---

**END OF AUDIT REPORT**
