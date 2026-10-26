"""
ner_evaluation.py
================
Entity-level evaluation for NER systems.
Compares rule-based predictions against ground-truth annotations.
"""

from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class Entity:
    """Represents a single entity annotation."""
    text: str
    label: str
    start: int
    end: int
    
    def __eq__(self, other):
        """Check exact match (text, label, start, end)."""
        if not isinstance(other, Entity):
            return False
        return (self.text == other.text and 
                self.label == other.label and 
                self.start == other.start and 
                self.end == other.end)
    
    def __hash__(self):
        return hash((self.text, self.label, self.start, self.end))
    
    @classmethod
    def from_dict(cls, entity_dict: Dict) -> 'Entity':
        """Create Entity from dictionary."""
        return cls(
            text=entity_dict['text'],
            label=entity_dict['label'],
            start=entity_dict['start'],
            end=entity_dict['end']
        )

class NEREvaluator:
    """Evaluates NER predictions against ground-truth annotations."""
    
    def __init__(self):
        """Initialize NER evaluator."""
        self.results = None

    @staticmethod
    def _normalize_entities(entities: List[Entity]) -> List[Entity]:
        """Convert entity dictionaries to hashable Entity objects."""
        normalized = []
        for entity in entities:
            if isinstance(entity, Entity):
                normalized.append(entity)
            elif isinstance(entity, dict):
                normalized.append(Entity.from_dict(entity))
            else:
                raise TypeError(
                    "Entities must be Entity instances or dictionaries with "
                    "text, label, start, and end fields"
                )
        return normalized
    
    def parse_entities_from_annotations(
        self,
        annotations: List[Dict]
    ) -> Dict[str, List[Entity]]:
        """
        Parse entities from annotation data.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Dictionary mapping record_id to list of Entity objects
        """
        entities_by_record = {}
        
        for annotation in annotations:
            record_id = annotation['record_id']
            entity = Entity.from_dict({
                'text': annotation['entity_text'],
                'label': annotation['entity_label'],
                'start': annotation['start_offset'],
                'end': annotation['end_offset']
            })
            
            if record_id not in entities_by_record:
                entities_by_record[record_id] = []
            
            entities_by_record[record_id].append(entity)
        
        return entities_by_record
    
    def parse_entities_from_annotation_dicts(
        self,
        annotation_dicts: Dict[str, List[Dict]]
    ) -> Dict[str, List[Entity]]:
        """
        Parse entities from annotation dictionary format (from AnnotationManager.parse_entities_from_annotations).
        
        Args:
            annotation_dicts: Dictionary mapping record_id to list of entity dicts
            
        Returns:
            Dictionary mapping record_id to list of Entity objects
        """
        entities_by_record = {}
        
        for record_id, entity_dicts in annotation_dicts.items():
            entities = [Entity.from_dict(e) for e in entity_dicts]
            entities_by_record[record_id] = entities
        
        return entities_by_record
    
    def parse_entities_from_predictions(
        self,
        predictions: List[Tuple[str, str]],
        record_text: str
    ) -> List[Entity]:
        """
        Parse entities from BIO-tagged predictions.
        
        Args:
            predictions: List of (token, label) tuples
            record_text: Original text for offset calculation
            
        Returns:
            List of Entity objects
        """
        entities = []
        current_entity = None
        current_start = 0
        text_position = 0
        
        for token, label in predictions:
            token_length = len(token)
            
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity is not None:
                    entities.append(Entity(
                        text=current_entity['text'],
                        label=current_entity['label'],
                        start=current_entity['start'],
                        end=text_position
                    ))
                
                # Start new entity
                entity_label = label.split('-')[1]
                current_entity = {
                    'text': token,
                    'label': entity_label,
                    'start': text_position
                }
                current_start = text_position
            
            elif label.startswith('I-') and current_entity is not None:
                # Continue current entity
                current_entity['text'] += ' ' + token
                # Note: We're using space-separated tokens, so position is approximate
                current_start = text_position
            
            elif label == 'O':
                # Save previous entity if exists
                if current_entity is not None:
                    entities.append(Entity(
                        text=current_entity['text'],
                        label=current_entity['label'],
                        start=current_entity['start'],
                        end=text_position
                    ))
                    current_entity = None
            
            text_position += token_length + 1  # +1 for space
        
        # Save last entity if exists
        if current_entity is not None:
            entities.append(Entity(
                text=current_entity['text'],
                label=current_entity['label'],
                start=current_entity['start'],
                end=text_position
            ))
        
        return entities

    def parse_entities_from_structured_text(
        self,
        record_text: str,
        field_label_mapping: Dict[str, str]
    ) -> List[Entity]:
        """Extract exact value spans for mapped fields in a linearized record."""
        entities = []
        cursor = 0

        for field_value in record_text.split(" | "):
            field_name, separator, value = field_value.partition(":")
            if separator:
                normalized_field = "".join(
                    character
                    for character in field_name.casefold()
                    if character.isalnum()
                )
                label = field_label_mapping.get(normalized_field)
                stripped_value = value.strip()
                if label and stripped_value.casefold() not in {"", "nan", "none"}:
                    value_offset = cursor + len(field_name) + 1
                    value_offset += len(value) - len(value.lstrip())
                    end_offset = value_offset + len(stripped_value)
                    entities.append(Entity(
                        text=record_text[value_offset:end_offset],
                        label=label,
                        start=value_offset,
                        end=end_offset
                    ))
            cursor += len(field_value) + len(" | ")

        return entities

    def parse_entities_from_improved_detection(
        self,
        record_text: str
    ) -> List[Entity]:
        """Use improved PII detection from data_pipeline for better accuracy."""
        try:
            from data_pipeline import improved_pii_detection
            detected = improved_pii_detection(record_text)
            entities = []
            for entity in detected:
                entities.append(Entity(
                    text=entity["text"],
                    label=entity["label"],
                    start=entity["start"],
                    end=entity["end"]
                ))
            return entities
        except ImportError:
            # Fallback to structured text parsing
            field_label_mapping = {
                "name": "PERSON",
                "village": "LOCATION",
                "villageestate": "LOCATION",
                "admissionnumber": "PATIENT_ID",
                "birthnotificationnumber": "PATIENT_ID",
                "dateofadmission": "DATE",
                "dateofdelivery": "DATE",
                "date": "DATE"
            }
            return self.parse_entities_from_structured_text(record_text, field_label_mapping)
    
    def evaluate_record(
        self,
        predicted_entities: List[Entity],
        reference_entities: List[Entity]
    ) -> Dict:
        """
        Evaluate entities for a single record.
        
        Args:
            predicted_entities: List of predicted Entity objects
            reference_entities: List of ground-truth Entity objects
            
        Returns:
            Dictionary with evaluation metrics
        """
        predicted_entities = self._normalize_entities(predicted_entities)
        reference_entities = self._normalize_entities(reference_entities)

        # Convert to sets for comparison
        predicted_set = set(predicted_entities)
        reference_set = set(reference_entities)
        
        # Calculate TP, FP, FN
        true_positives = predicted_set & reference_set
        false_positives = predicted_set - reference_set
        false_negatives = reference_set - predicted_set
        
        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)
        tn = 0  # Not applicable for entity-level evaluation
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_count': len(predicted_entities),
            'reference_count': len(reference_entities)
        }
    
    def evaluate_per_entity(
        self,
        predicted_entities: List[Entity],
        reference_entities: List[Entity]
    ) -> Dict[str, Dict]:
        """
        Evaluate per-entity-type metrics.
        
        Args:
            predicted_entities: List of predicted Entity objects
            reference_entities: List of ground-truth Entity objects
            
        Returns:
            Dictionary mapping entity labels to metrics
        """
        predicted_entities = self._normalize_entities(predicted_entities)
        reference_entities = self._normalize_entities(reference_entities)

        # Group by entity type
        predicted_by_type = {}
        reference_by_type = {}
        
        for entity in predicted_entities:
            if entity.label not in predicted_by_type:
                predicted_by_type[entity.label] = []
            predicted_by_type[entity.label].append(entity)
        
        for entity in reference_entities:
            if entity.label not in reference_by_type:
                reference_by_type[entity.label] = []
            reference_by_type[entity.label].append(entity)
        
        # Get all entity types
        all_types = set(predicted_by_type.keys()) | set(reference_by_type.keys())
        
        results = {}
        
        for entity_type in all_types:
            preds = predicted_by_type.get(entity_type, [])
            refs = reference_by_type.get(entity_type, [])
            
            result = self.evaluate_record(preds, refs)
            results[entity_type] = {
                'true_positives': result['true_positives'],
                'false_positives': result['false_positives'],
                'false_negatives': result['false_negatives'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'predicted_count': result['predicted_count'],
                'reference_count': result['reference_count']
            }
        
        return results
    
    def evaluate_dataset(
        self,
        predicted_entities_by_record: Dict[str, List[Entity]],
        reference_entities_by_record: Dict[str, List[Entity]]
    ) -> Dict:
        """
        Evaluate across all records.
        
        Args:
            predicted_entities_by_record: Dict mapping record_id to predicted entities
            reference_entities_by_record: Dict mapping record_id to reference entities
            
        Returns:
            Dictionary with overall and per-entity metrics
        """
        if not reference_entities_by_record:
            raise ValueError("No reference annotations available for evaluation")
        
        # Get common records
        common_records = set(predicted_entities_by_record.keys()) & set(reference_entities_by_record.keys())
        
        if not common_records:
            raise ValueError("No common records between predictions and annotations")
        
        # Aggregate per-entity metrics
        all_tp = 0
        all_fp = 0
        all_fn = 0
        
        per_entity_aggregates = {}
        
        for record_id in common_records:
            preds = predicted_entities_by_record.get(record_id, [])
            refs = reference_entities_by_record.get(record_id, [])
            
            # Overall metrics
            result = self.evaluate_record(preds, refs)
            all_tp += result['true_positives']
            all_fp += result['false_positives']
            all_fn += result['false_negatives']
            
            # Per-entity metrics
            per_entity = self.evaluate_per_entity(preds, refs)
            for entity_type, metrics in per_entity.items():
                if entity_type not in per_entity_aggregates:
                    per_entity_aggregates[entity_type] = {
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0
                    }
                per_entity_aggregates[entity_type]['true_positives'] += metrics['true_positives']
                per_entity_aggregates[entity_type]['false_positives'] += metrics['false_positives']
                per_entity_aggregates[entity_type]['false_negatives'] += metrics['false_negatives']
        
        # Calculate overall metrics
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Calculate per-entity metrics
        per_entity_metrics = {}
        for entity_type, aggregates in per_entity_aggregates.items():
            tp = aggregates['true_positives']
            fp = aggregates['false_positives']
            fn = aggregates['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_entity_metrics[entity_type] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        self.results = {
            'overall': {
                'true_positives': all_tp,
                'false_positives': all_fp,
                'false_negatives': all_fn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'records_evaluated': len(common_records)
            },
            'per_entity': per_entity_metrics,
            'evaluated_record_ids': sorted(common_records),
            'matching_method': 'exact_span_and_type',
            'description': 'Prediction is TP only when start offset, end offset, and entity type all match reference'
        }
        
        return self.results
    
    def get_error_analysis(
        self,
        predicted_entities: List[Entity],
        reference_entities: List[Entity]
    ) -> Dict:
        """
        Analyze prediction errors.
        
        Args:
            predicted_entities: List of predicted Entity objects
            reference_entities: List of ground-truth Entity objects
            
        Returns:
            Dictionary with error analysis
        """
        predicted_entities = self._normalize_entities(predicted_entities)
        reference_entities = self._normalize_entities(reference_entities)

        predicted_set = set(predicted_entities)
        reference_set = set(reference_entities)
        
        false_positives = predicted_set - reference_set
        false_negatives = reference_set - predicted_set
        
        return {
            'false_positives': [
                {'text': e.text, 'label': e.label, 'start': e.start, 'end': e.end}
                for e in false_positives
            ],
            'false_negatives': [
                {'text': e.text, 'label': e.label, 'start': e.start, 'end': e.end}
                for e in false_negatives
            ],
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives)
        }
    
    def save_results(self, output_path: Path) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Output file path
        """
        if self.results is None:
            raise ValueError("No evaluation results to save. Run evaluate_dataset first.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
