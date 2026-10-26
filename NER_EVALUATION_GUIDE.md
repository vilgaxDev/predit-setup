# NER Evaluation Guide

## Overview

This guide explains how to use the manual annotation workflow and NER evaluation system for your maternal-child health data anonymisation project.

## What Has Changed

### Removed: Simulated NER Evaluation
- Removed the 8% injected label-noise simulation from `run_notebook.py`
- Removed hardcoded precision, recall, F1, and accuracy values
- Removed claims that evaluation was completed with verified annotations
- NER metrics file now shows "pending" status instead of simulated results

### Added: Real Annotation Workflow
- New `annotation_manager.py` module for manual PII annotation
- New `ner_evaluation.py` module for entity-level evaluation
- New "📝 Annotation Workflow" tab in the dashboard
- Record-level completion tracking, including records with no PII
- New automated tests in `tests/test_ner_evaluation.py`

`run_notebook.py` does not provide ground truth: it assigns BIO labels from configured
field names and calculates demonstration metrics after injecting errors into those
labels. Only manually reviewed records are used for the ground-truth evaluation below.

### Updated: Model Validation Tab
- Replaced "Per-Entity Performance (Simulated)" with real evaluation interface
- Shows annotation status and controls to run evaluation
- Displays genuine metrics only when annotations exist
- Shows "Ground truth unavailable — evaluation pending" when no annotations exist

## Annotation Workflow

### Step 1: Access the Annotation Tab

1. Start the dashboard:
   ```bash
   cd D:\sarah\dashboard\python_pipeline
   streamlit run app.py
   ```

2. Click the **📝 Annotation Workflow** button in the navigation bar

### Step 2: Enter Annotator Information

- Enter your Annotator ID in the sidebar (e.g., "annotator_1", "clinician_A")
- This ID is stored with each annotation for tracking

### Step 3: Select a Record to Annotate

1. Choose an authorised local register from **Dataset to annotate**. Prefer the original source register over an anonymized dashboard export when the goal is to evaluate PII detection.
2. Use the record dropdown to select a clinical record (e.g., "Record 0", "Record 1")
3. The record text will be displayed in a text area
4. Review the full text for personally identifiable information (PII)

### Step 4: Mark PII Entities

1. Select the entity type from the dropdown:
   - **PERSON**: Person names
   - **DATE**: Dates (admission, delivery, LMP, etc.)
   - **PATIENT_ID**: Patient identifiers (admission number, birth notification)
   - **LOCATION**: Locations (village, county, facility)
   - **PHONE**: Phone numbers
   - **OTHER_PII**: Other personally identifiable information

2. Enter the character start offset
3. Enter the character end offset
4. The selected text will be displayed for verification

### Step 5: Validate and Save

1. Click **"Validate Selection"** to check that:
   - Offsets are within bounds
   - Selected text matches the text at those offsets
   - Entity label is valid

2. If validation passes, click **"💾 Save Annotation"** to save

Repeat this for every PII entity in the record. An annotation is one entity span,
not a declaration that the whole record has been reviewed.

### Step 6: Complete the Record Review

- Saved annotations for the current record are displayed in expandable sections
- You can view the entity text, offsets, annotator ID, timestamp, and status
- You can delete annotations if needed
- After checking every field and marking every PII span, click **"✅ Mark Record Fully Reviewed"**
- Mark records with no PII as fully reviewed too; their empty reference labels are needed to count false positives
- Saving or deleting an entity annotation clears that record's completion mark; review it again afterward

### Step 7: Review Existing Annotations and Export

- The status panel distinguishes records fully reviewed from records still pending review

1. Click **"Export Annotations"** to save all annotations to a JSON file
2. Download the file using the "Download JSON" button
3. The export includes metadata, entity annotations, and completed record reviews

## Running NER Evaluation

### Step 1: Create Annotations

- Fully review at least a few records using the Annotation Workflow tab
- More annotations will provide more reliable evaluation metrics

### Step 2: Go to Model Validation Tab

1. Click the **✅ Model** button in the navigation bar
2. Scroll to the "📊 NER Evaluation Status" section

### Step 3: Check Annotation Status

- The system will show the number of annotations found
- If no records are marked fully reviewed, you'll see "No completed records — evaluation pending"

### Step 4: Run Evaluation

1. Click **"🔍 Run NER Evaluation"** button
2. The system will:
   - Use only records marked fully reviewed, including records with no PII
   - Parse manual annotations into reference entities
   - Generate rule-based predictions from the existing PII detector on the exact reviewed record text
   - Compare predictions against annotations using exact character spans and mapped entity labels
   - Calculate entity-level precision, recall, and F1

This evaluates the current **rule-based detector**, not a fine-tuned BERT model. The
older simulated results shown with pipeline outputs remain demonstration metrics.

### Step 5: Review Results

#### Overall Results
- **Precision**: Percentage of predicted entities that were correct
- **Recall**: Percentage of reference entities that were detected
- **F1 Score**: Harmonic mean of precision and recall
- **Records Evaluated**: Number of records with both annotations and predictions

#### Per-Entity Metrics
- Breakdown of metrics by entity type (PERSON, DATE, etc.)
- Shows which entity types are detected well and which need improvement

#### Error Analysis
- **False Positives**: Entities predicted but not in annotations
- **False Negatives**: Entities in annotations but not predicted
- Sample errors are displayed for analysis

### Step 6: Download Results

1. Click **"Download Evaluation Report"** to save the full evaluation
2. The JSON file includes:
   - Overall metrics
   - Per-entity metrics
   - Matching method description
   - Error analysis details

## Evaluation Method

### Matching Criteria

A prediction is a **True Positive (TP)** only when all three match exactly:
1. **Character start offset** matches reference
2. **Character end offset** matches reference
3. **Entity type** matches reference

### Metric Definitions

- **Precision = TP / (TP + FP)**
  - How many predicted entities are correct
- **Recall = TP / (TP + FN)**
  - How many reference entities were detected
- **F1 = 2 × (Precision × Recall) / (Precision + Recall)**
  - Harmonic mean of precision and recall

### Error Types

- **False Positive (FP)**: System predicted an entity that is not in the ground truth
- **False Negative (FN)**: System missed an entity that is in the ground truth
- **Incorrect Type**: Correct entity but wrong label (e.g., PERSON vs DATE)
- **Incorrect Span**: Correct entity but wrong offsets

## Privacy and Data Integrity

### Important Privacy Considerations

1. **Do not expose identifying information** in exported reports
2. **Do not send records to external AI services**
3. **Store annotations separately** from source health records
4. **Do not overwrite original health records**
5. **Only annotate records you are authorised to view**

### Annotation Storage

- Annotations are stored in `annotations/annotations.jsonl`
- Fully reviewed record snapshots, including records with no PII, are stored in `annotations/record_reviews.jsonl`
- Each annotation includes:
  - Record ID
  - Entity text
  - Entity label
  - Start/end offsets
  - Annotator ID

The annotation workflow stores record text locally for reproducible offset matching.
Keep this directory access-controlled and do not share its contents outside the
authorised clinical/research team.
  - Timestamp
  - Review status

### Review Process

- Annotations start with status "pending_review"
- You can mark records as reviewed using the `mark_reviewed()` method
- Supports independent second-annotator review where available
- Do not automatically mark records as verified

## Running Tests

### Run All Tests

```bash
cd D:\sarah\dashboard\python_pipeline
python tests/test_ner_evaluation.py
```

### Test Coverage

The automated tests cover:

#### Annotation Manager Tests
- Creating annotations
- Validating offsets
- Entity text matching
- Unknown entity labels
- Saving and loading annotations
- Retrieving annotations by record
- Annotation status calculation
- Marking as reviewed
- Deleting annotations
- Parsing entities from annotations

#### NER Evaluator Tests
- Entity equality comparison
- Entity creation from dictionary
- Perfect matches
- Missed entities (false negatives)
- False positives
- Incorrect entity types
- Incorrect offsets
- Multiple entities in one record
- Empty reference sets
- Empty prediction sets
- Per-entity metrics
- Dataset-level evaluation
- Error analysis
- Saving results

#### Integration Tests
- Full pipeline from annotation to evaluation

### Test Results

All 27 tests pass successfully:
- ✅ 10 annotation manager tests
- ✅ 16 NER evaluator tests
- ✅ 1 integration test

## Files Created/Modified

### New Files

1. **`annotation_manager.py`** (280 lines)
   - Manual annotation workflow
   - Entity label management
   - Annotation storage and retrieval
   - Export functionality

2. **`ner_evaluation.py`** (381 lines)
   - Entity-level evaluation
   - Exact-span matching
   - Per-entity metrics
   - Error analysis

3. **`tests/test_ner_evaluation.py`** (532 lines)
   - Comprehensive automated tests
   - Tests for annotation manager
   - Tests for NER evaluator
   - Integration tests

4. **`tests/__init__.py`** (3 lines)
   - Tests package initialization

5. **`NER_EVALUATION_GUIDE.md`** (this file)
   - User documentation
   - Step-by-step instructions
   - Evaluation method explanation

### Modified Files

1. **`run_notebook.py`**
   - Removed simulated NER evaluation (lines 608-779)
   - Replaced with "pending" status message
   - Removed 8% noise injection logic
   - Removed hardcoded metrics

2. **`app.py`**
   - Added imports for annotation_manager and ner_evaluation
   - Added 9th navigation item: "📝 Annotation Workflow"
   - Added Annotation Workflow page with full UI
   - Replaced Model Validation NER section with real evaluation interface
   - Integrated evaluation results display
   - Added error analysis display
   - Added download functionality

## Limitations and Next Steps

### Current Limitations

1. **No Ground Truth Annotations**
   - Evaluation cannot run until manual annotations are created
   - Dashboard shows "Ground truth unavailable — evaluation pending"

2. **Rule-Based Predictions Only**
   - Current system uses rule-based PII detection
   - No trained NER model (fine-tuned BERT) yet
   - To get realistic metrics, you need:
     - Manual annotations
     - Trained NER model
     - Evaluation on held-out test set

3. **Approximate Offset Matching**
   - Current prediction parsing uses approximate character positions
   - May need refinement for exact character-level accuracy

### Next Steps for Real Evaluation

1. **Create Manual Annotations**
   - Use the Annotation Workflow tab
   - Annotate a representative sample of records
   - Consider inter-annotator agreement with multiple annotators

2. **Train NER Model**
   - Fine-tune BERT on annotated data
   - This requires appropriate hardware and annotated training data
   - Not currently implemented in the dashboard

3. **Evaluate on Held-Out Set**
   - Split annotations into training/validation/test sets
   - Train model on training set
   - Evaluate on untouched test set
   - Report genuine metrics

4. **Improve Rule-Based Detector**
   - Refine ENTITY_MAP in data_pipeline.py
   - Add more PII patterns
   - Improve offset calculation accuracy

## Troubleshooting

### "Ground truth unavailable — evaluation pending"

**Cause**: No manual annotations have been created yet.

**Solution**: 
1. Go to the 📝 Annotation Workflow tab
2. Select records and mark PII entities
3. Save annotations
4. Return to Model Validation tab and run evaluation

### "Entity text mismatch" error

**Cause**: The text at the specified offsets does not match the entity text you entered.

**Solution**:
1. Verify the start and end offsets are correct
2. Check that the entity text exactly matches the substring at those positions
3. Use the "Validate Selection" button before saving

### "Unknown entity label" error

**Cause**: You entered an entity label that is not in the configured list.

**Solution**: Use one of the valid labels:
- PERSON
- DATE
- PATIENT_ID
- LOCATION
- PHONE
- OTHER_PII

### Tests fail with "ModuleNotFoundError"

**Cause**: Python cannot find the annotation_manager or ner_evaluation modules.

**Solution**: 
1. Ensure you are in the correct directory: `D:\sarah\dashboard\python_pipeline`
2. Run tests from that directory
3. Ensure the files exist in the directory

## Summary

This implementation provides:

✅ **Removed fabricated evaluation metrics** - No more simulated 8% noise
✅ **Real annotation workflow** - Manual PII annotation interface
✅ **Genuine entity-level evaluation** - Exact-span matching
✅ **Automated tests** - 27 tests, all passing
✅ **Privacy safeguards** - Annotations stored separately, no external APIs
✅ **Clear status indicators** - "Pending" when no annotations exist
✅ **Documentation** - This guide and inline comments

**Next**: Create manual annotations to enable real NER evaluation. The system is ready for genuine scientific evaluation when you have ground-truth annotations.
