"""
test_ner_evaluation.py
=======================
Automated tests for NER evaluation system.
Tests annotation management, entity-level evaluation, and metric calculations.
"""

import pytest
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation_manager import AnnotationManager
from ner_evaluation import NEREvaluator, Entity


class TestAnnotationManager:
    """Test annotation manager functionality."""
    
    def test_create_annotation(self):
        """Test creating a single annotation."""
        manager = AnnotationManager()
        
        annotation = manager.create_annotation(
            record_id="test_record_1",
            record_text="Name: John Doe | Age: 30",
            entity_text="John Doe",
            entity_label="PERSON",
            start_offset=6,
            end_offset=14,
            annotator_id="test_annotator"
        )
        
        assert annotation['record_id'] == "test_record_1"
        assert annotation['entity_text'] == "John Doe"
        assert annotation['entity_label'] == "PERSON"
        assert annotation['start_offset'] == 6
        assert annotation['end_offset'] == 14
        assert annotation['annotator_id'] == "test_annotator"
        assert annotation['status'] == "pending_review"
    
    def test_invalid_offsets(self):
        """Test that invalid offsets raise ValueError."""
        manager = AnnotationManager()
        
        with pytest.raises(ValueError, match="Invalid offsets"):
            manager.create_annotation(
                record_id="test_record",
                record_text="Name: John",
                entity_text="John",
                entity_label="PERSON",
                start_offset=-1,
                end_offset=10,
                annotator_id="test"
            )
        
        with pytest.raises(ValueError, match="Invalid offsets"):
            manager.create_annotation(
                record_id="test_record",
                record_text="Name: John",
                entity_text="John",
                entity_label="PERSON",
                start_offset=0,
                end_offset=100,
                annotator_id="test"
            )
    
    def test_entity_text_mismatch(self):
        """Test that entity text mismatch raises ValueError."""
        manager = AnnotationManager()
        
        with pytest.raises(ValueError, match="Entity text mismatch"):
            manager.create_annotation(
                record_id="test_record",
                record_text="Name: John",
                entity_text="Jane",
                entity_label="PERSON",
                start_offset=6,
                end_offset=10,
                annotator_id="test"
            )
    
    def test_unknown_entity_label(self):
        """Test that unknown entity labels raise ValueError."""
        manager = AnnotationManager()
        
        with pytest.raises(ValueError, match="Unknown entity label"):
            manager.create_annotation(
                record_id="test_record",
                record_text="Name: John",
                entity_text="John",
                entity_label="INVALID_LABEL",
                start_offset=6,
                end_offset=10,
                annotator_id="test"
            )
    
    def test_save_and_load_annotation(self, tmp_path):
        """Test saving and loading annotations."""
        manager = AnnotationManager(annotations_dir=tmp_path)
        
        annotation = manager.create_annotation(
            record_id="test_record",
            record_text="Name: John Doe",
            entity_text="John Doe",
            entity_label="PERSON",
            start_offset=6,
            end_offset=14,
            annotator_id="test"
        )
        
        manager.save_annotation(annotation)
        
        loaded = manager.load_annotations()
        assert len(loaded) == 1
        assert loaded[0]['entity_text'] == "John Doe"
    
    def test_get_annotations_by_record(self, tmp_path):
        """Test retrieving annotations by record ID."""
        manager = AnnotationManager(annotations_dir=tmp_path)
        
        # Create annotations for two different records
        ann1 = manager.create_annotation(
            record_id="record_1",
            record_text="Name: John",
            entity_text="John",
            entity_label="PERSON",
            start_offset=6,
            end_offset=10,
            annotator_id="test"
        )
        
        ann2 = manager.create_annotation(
            record_id="record_2",
            record_text="Name: Jane",
            entity_text="Jane",
            entity_label="PERSON",
            start_offset=6,
            end_offset=10,
            annotator_id="test"
        )
        
        manager.save_annotation(ann1)
        manager.save_annotation(ann2)
        
        record_1_anns = manager.get_annotations_by_record("record_1")
        assert len(record_1_anns) == 1
        assert record_1_anns[0]['entity_text'] == "John"
    
    def test_annotation_status(self, tmp_path):
        """Test annotation status calculation."""
        manager = AnnotationManager(annotations_dir=tmp_path)
        
        ann1 = manager.create_annotation(
            record_id="record_1",
            record_text="Name: John",
            entity_text="John",
            entity_label="PERSON",
            start_offset=6,
            end_offset=10,
            annotator_id="test"
        )
        
        manager.save_annotation(ann1)
        
        status = manager.get_annotation_status()
        assert status['total_records'] == 1
        assert status['total_annotations'] == 1
        assert status['reviewed'] == 0
        assert status['pending'] == 1
        assert status['completed_records'] == 0
        assert status['pending_records'] == 1

    def test_complete_record_review_includes_records_without_pii(self, tmp_path):
        """A fully reviewed record with no entities is still evaluation data."""
        manager = AnnotationManager(annotations_dir=tmp_path)

        manager.complete_record_review(
            record_id="record_1",
            record_text="Outcome: routine",
            annotator_id="clinician_1"
        )

        reviews = manager.load_record_reviews()
        status = manager.get_annotation_status()
        assert reviews[0]["record_id"] == "record_1"
        assert reviews[0]["record_text"] == "Outcome: routine"
        assert status["total_records"] == 1
        assert status["total_annotations"] == 0
        assert status["completed_records"] == 1
        assert status["completion_rate"] == 1.0

    def test_saving_annotation_invalidates_completed_review(self, tmp_path):
        """Adding or correcting an entity requires the record to be reviewed again."""
        manager = AnnotationManager(annotations_dir=tmp_path)
        manager.complete_record_review("record_1", "Name: John", "annotator_1")
        annotation = manager.create_annotation(
            record_id="record_1",
            record_text="Name: John",
            entity_text="John",
            entity_label="PERSON",
            start_offset=6,
            end_offset=10,
            annotator_id="annotator_1"
        )

        manager.save_annotation(annotation)

        assert manager.load_record_reviews() == []
        assert manager.get_annotation_status()["pending_records"] == 1
    
    def test_mark_reviewed(self, tmp_path):
        """Test marking annotations as reviewed."""
        manager = AnnotationManager(annotations_dir=tmp_path)
        
        ann1 = manager.create_annotation(
            record_id="record_1",
            record_text="Name: John",
            entity_text="John",
            entity_label="PERSON",
            start_offset=6,
            end_offset=10,
            annotator_id="test"
        )
        
        manager.save_annotation(ann1)
        manager.mark_reviewed("record_1", reviewer_id="reviewer_1")
        
        loaded = manager.load_annotations()
        assert loaded[0]['status'] == "reviewed"
        assert loaded[0]['reviewed_by'] == "reviewer_1"
    
    def test_delete_annotation(self, tmp_path):
        """Test deleting an annotation."""
        manager = AnnotationManager(annotations_dir=tmp_path)
        
        ann1 = manager.create_annotation(
            record_id="record_1",
            record_text="Name: John",
            entity_text="John",
            entity_label="PERSON",
            start_offset=6,
            end_offset=10,
            annotator_id="test"
        )
        
        manager.save_annotation(ann1)
        
        deleted = manager.delete_annotation("record_1", "John", 6)
        assert deleted is True
        
        loaded = manager.load_annotations()
        assert len(loaded) == 0
    
    def test_parse_entities_from_annotations(self):
        """Test static method for parsing entities from annotations."""
        annotations = [
            {
                'record_id': 'record_1',
                'entity_text': 'John Doe',
                'entity_label': 'PERSON',
                'start_offset': 6,
                'end_offset': 14
            }
        ]
        
        result = AnnotationManager.parse_entities_from_annotations(annotations)
        assert 'record_1' in result
        assert len(result['record_1']) == 1
        assert result['record_1'][0]['text'] == 'John Doe'
        assert result['record_1'][0]['label'] == 'PERSON'


class TestNEREvaluator:
    """Test NER evaluation functionality."""

    def test_prediction_parser_uses_exact_spans_and_annotation_labels(self):
        evaluator = NEREvaluator()
        text = "Name: John Doe | Village: Kisumu"
        predictions = [
            ("Name:", "O"),
            ("John", "B-NAME"),
            ("Doe", "I-NAME"),
            ("|", "O"),
            ("Village:", "O"),
            ("Kisumu", "B-VILLAGE")
        ]

        entities = evaluator.parse_entities_from_predictions(
            predictions,
            text,
            label_mapping={"NAME": "PERSON", "VILLAGE": "LOCATION"}
        )

        assert entities == [
            Entity(text="John Doe", label="PERSON", start=6, end=14),
            Entity(
                text="Kisumu",
                label="LOCATION",
                start=text.index("Kisumu"),
                end=text.index("Kisumu") + len("Kisumu")
            )
        ]

    def test_prediction_parser_ignores_unmapped_entity_labels(self):
        evaluator = NEREvaluator()

        entities = evaluator.parse_entities_from_predictions(
            [("HIV:", "O"), ("positive", "B-HIV_STATUS")],
            "HIV: positive",
            label_mapping={"NAME": "PERSON"}
        )

        assert entities == []

    def test_prediction_parser_rejects_tokens_not_in_source_text(self):
        evaluator = NEREvaluator()

        with pytest.raises(ValueError, match="was not found"):
            evaluator.parse_entities_from_predictions(
                [("Unknown", "B-NAME")],
                "Name: John"
            )

    def test_structured_prediction_parser_extracts_only_mapped_field_values(self):
        evaluator = NEREvaluator()
        text = (
            "Name: Jane Doe | Village: Nairobi | Age: 29 | "
            "Date of Delivery: 21/01/2014 | HIV Status: N"
        )

        entities = evaluator.parse_entities_from_structured_text(
            text,
            {
                "name": "PERSON",
                "village": "LOCATION",
                "dateofdelivery": "DATE"
            }
        )

        assert [(entity.text, entity.label) for entity in entities] == [
            ("Jane Doe", "PERSON"),
            ("Nairobi", "LOCATION"),
            ("21/01/2014", "DATE")
        ]
        for entity in entities:
            assert text[entity.start:entity.end] == entity.text

    def test_dataset_evaluation_counts_fully_reviewed_record_without_pii(self):
        evaluator = NEREvaluator()

        results = evaluator.evaluate_dataset(
            {"record_without_pii": []},
            {"record_without_pii": []}
        )

        assert results["overall"]["records_evaluated"] == 1
        assert results["overall"]["true_positives"] == 0
        assert results["overall"]["false_positives"] == 0
        assert results["overall"]["false_negatives"] == 0
    
    def test_entity_equality(self):
        """Test Entity equality comparison."""
        entity1 = Entity(text="John", label="PERSON", start=0, end=4)
        entity2 = Entity(text="John", label="PERSON", start=0, end=4)
        entity3 = Entity(text="Jane", label="PERSON", start=0, end=4)
        
        assert entity1 == entity2
        assert entity1 != entity3
    
    def test_entity_from_dict(self):
        """Test creating Entity from dictionary."""
        entity_dict = {
            'text': 'John',
            'label': 'PERSON',
            'start': 0,
            'end': 4
        }
        
        entity = Entity.from_dict(entity_dict)
        assert entity.text == 'John'
        assert entity.label == 'PERSON'
        assert entity.start == 0
        assert entity.end == 4
    
    def test_perfect_match(self):
        """Test evaluation with perfect match."""
        evaluator = NEREvaluator()
        
        predicted = [Entity(text="John", label="PERSON", start=0, end=4)]
        reference = [Entity(text="John", label="PERSON", start=0, end=4)]
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['true_positives'] == 1
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1'] == 1.0
    
    def test_missed_entity(self):
        """Test evaluation with missed entity (false negative)."""
        evaluator = NEREvaluator()
        
        predicted = []
        reference = [Entity(text="John", label="PERSON", start=0, end=4)]
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['true_positives'] == 0
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 1
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1'] == 0.0
    
    def test_false_positive(self):
        """Test evaluation with false positive."""
        evaluator = NEREvaluator()
        
        predicted = [Entity(text="John", label="PERSON", start=0, end=4)]
        reference = []
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['true_positives'] == 0
        assert result['false_positives'] == 1
        assert result['false_negatives'] == 0
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0  # No reference entities, so recall is 0
        assert result['f1'] == 0.0
    
    def test_incorrect_entity_type(self):
        """Test evaluation with incorrect entity type."""
        evaluator = NEREvaluator()
        
        predicted = [Entity(text="John", label="PERSON", start=0, end=4)]
        reference = [Entity(text="John", label="DATE", start=0, end=4)]
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['true_positives'] == 0
        assert result['false_positives'] == 1
        assert result['false_negatives'] == 1
    
    def test_incorrect_offsets(self):
        """Test evaluation with incorrect offsets."""
        evaluator = NEREvaluator()
        
        predicted = [Entity(text="John", label="PERSON", start=0, end=4)]
        reference = [Entity(text="John", label="PERSON", start=5, end=9)]
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['true_positives'] == 0
        assert result['false_positives'] == 1
        assert result['false_negatives'] == 1
    
    def test_multiple_entities(self):
        """Test evaluation with multiple entities."""
        evaluator = NEREvaluator()
        
        predicted = [
            Entity(text="John", label="PERSON", start=0, end=4),
            Entity(text="2023", label="DATE", start=10, end=14)
        ]
        reference = [
            Entity(text="John", label="PERSON", start=0, end=4),
            Entity(text="2023", label="DATE", start=10, end=14)
        ]
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['true_positives'] == 2
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1'] == 1.0
    
    def test_empty_references(self):
        """Test evaluation with empty reference set."""
        evaluator = NEREvaluator()
        
        predicted = [Entity(text="John", label="PERSON", start=0, end=4)]
        reference = []
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0  # No reference entities, so recall is 0
    
    def test_empty_predictions(self):
        """Test evaluation with empty prediction set."""
        evaluator = NEREvaluator()
        
        predicted = []
        reference = [Entity(text="John", label="PERSON", start=0, end=4)]
        
        result = evaluator.evaluate_record(predicted, reference)
        
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
    
    def test_per_entity_metrics(self):
        """Test per-entity-type metrics."""
        evaluator = NEREvaluator()
        
        predicted = [
            Entity(text="John", label="PERSON", start=0, end=4),
            Entity(text="2023", label="DATE", start=10, end=14)
        ]
        reference = [
            Entity(text="John", label="PERSON", start=0, end=4),
            Entity(text="2024", label="DATE", start=10, end=14)
        ]
        
        per_entity = evaluator.evaluate_per_entity(predicted, reference)
        
        assert "PERSON" in per_entity
        assert "DATE" in per_entity
        assert per_entity["PERSON"]["true_positives"] == 1
        assert per_entity["DATE"]["true_positives"] == 0
        assert per_entity["DATE"]["false_negatives"] == 1
    
    def test_dataset_evaluation(self):
        """Test evaluation across multiple records."""
        evaluator = NEREvaluator()
        
        predicted_by_record = {
            "record_1": [Entity(text="John", label="PERSON", start=0, end=4)],
            "record_2": [Entity(text="Jane", label="PERSON", start=0, end=4)]
        }
        
        reference_by_record = {
            "record_1": [Entity(text="John", label="PERSON", start=0, end=4)],
            "record_2": [Entity(text="Jane", label="PERSON", start=0, end=4)]
        }
        
        results = evaluator.evaluate_dataset(predicted_by_record, reference_by_record)
        
        assert results['overall']['true_positives'] == 2
        assert results['overall']['false_positives'] == 0
        assert results['overall']['false_negatives'] == 0
        assert results['overall']['precision'] == 1.0
        assert results['overall']['recall'] == 1.0
        assert results['overall']['f1'] == 1.0
        assert results['overall']['records_evaluated'] == 2
        assert results['evaluated_record_ids'] == ["record_1", "record_2"]

    def test_evaluation_accepts_entity_dictionaries(self):
        """Test evaluation handles dictionaries returned by PII detectors."""
        evaluator = NEREvaluator()
        entity = {
            "text": "John",
            "label": "PERSON",
            "start": 0,
            "end": 4
        }

        results = evaluator.evaluate_dataset(
            {"record_1": [entity]},
            {"record_1": [entity.copy()]}
        )
        errors = evaluator.get_error_analysis([entity], [entity.copy()])

        assert results["overall"]["true_positives"] == 1
        assert results["overall"]["false_positives"] == 0
        assert results["overall"]["false_negatives"] == 0
        assert errors["false_positives"] == []
        assert errors["false_negatives"] == []
    
    def test_no_reference_annotations(self):
        """Test that evaluation raises error without reference annotations."""
        evaluator = NEREvaluator()
        
        with pytest.raises(ValueError, match="No reference annotations"):
            evaluator.evaluate_dataset({}, {})
    
    def test_no_common_records(self):
        """Test that evaluation raises error without common records."""
        evaluator = NEREvaluator()
        
        predicted_by_record = {"record_1": []}
        reference_by_record = {"record_2": []}
        
        with pytest.raises(ValueError, match="No common records"):
            evaluator.evaluate_dataset(predicted_by_record, reference_by_record)
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        evaluator = NEREvaluator()
        
        predicted = [
            Entity(text="John", label="PERSON", start=0, end=4),
            Entity(text="Fake", label="PERSON", start=10, end=14)
        ]
        reference = [
            Entity(text="John", label="PERSON", start=0, end=4),
            Entity(text="Real", label="PERSON", start=20, end=24)
        ]
        
        error_analysis = evaluator.get_error_analysis(predicted, reference)
        
        assert error_analysis['fp_count'] == 1
        assert error_analysis['fn_count'] == 1
        assert len(error_analysis['false_positives']) == 1
        assert len(error_analysis['false_negatives']) == 1
    
    def test_save_results(self, tmp_path):
        """Test saving evaluation results."""
        evaluator = NEREvaluator()
        
        predicted_by_record = {
            "record_1": [Entity(text="John", label="PERSON", start=0, end=4)]
        }
        reference_by_record = {
            "record_1": [Entity(text="John", label="PERSON", start=0, end=4)]
        }
        
        evaluator.evaluate_dataset(predicted_by_record, reference_by_record)
        
        output_path = tmp_path / "results.json"
        evaluator.save_results(output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert 'overall' in data
            assert 'per_entity' in data


class TestIntegration:
    """Integration tests for annotation and evaluation workflow."""
    
    def test_annotation_to_evaluation_pipeline(self, tmp_path):
        """Test full pipeline from annotation to evaluation."""
        # Create annotations
        manager = AnnotationManager(annotations_dir=tmp_path)
        
        ann1 = manager.create_annotation(
            record_id="record_1",
            record_text="Name: John Doe",
            entity_text="John Doe",
            entity_label="PERSON",
            start_offset=6,
            end_offset=14,
            annotator_id="test"
        )
        
        manager.save_annotation(ann1)
        
        # Load annotations
        annotations = manager.load_annotations()
        
        # Parse for evaluation
        annotation_dicts = AnnotationManager.parse_entities_from_annotations(annotations)
        
        # Evaluate
        evaluator = NEREvaluator()
        reference_entities = evaluator.parse_entities_from_annotation_dicts(annotation_dicts)
        
        # Create mock predictions
        predicted_entities = {
            "record_1": [Entity(text="John Doe", label="PERSON", start=6, end=14)]
        }
        
        results = evaluator.evaluate_dataset(predicted_entities, reference_entities)
        
        assert results['overall']['true_positives'] == 1
        assert results['overall']['precision'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
