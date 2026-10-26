"""
annotation_manager.py
===================
Manual annotation workflow for PII entities in clinical records.
Supports annotation, validation, and export for NER evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import hashlib

class AnnotationManager:
    """Manages manual annotations of PII entities in clinical records."""
    
    # Configurable entity labels (class attribute for easy access)
    ENTITY_LABELS = {
        "PERSON": "Person names",
        "DATE": "Dates (admission, delivery, LMP, etc.)",
        "PATIENT_ID": "Patient identifiers (admission number, birth notification)",
        "LOCATION": "Locations (village, county, facility)",
        "PHONE": "Phone numbers",
        "OTHER_PII": "Other personally identifiable information"
    }
    
    def __init__(self, annotations_dir: Path = None):
        """
        Initialize annotation manager.
        
        Args:
            annotations_dir: Directory to store annotation files
        """
        if annotations_dir is None:
            annotations_dir = Path(__file__).parent / "annotations"
        
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(exist_ok=True)
        
        self.annotations_file = self.annotations_dir / "annotations.jsonl"
        self.record_reviews_file = self.annotations_dir / "record_reviews.jsonl"
        self.metadata_file = self.annotations_dir / "metadata.json"
        
    def create_annotation(
        self,
        record_id: str,
        record_text: str,
        entity_text: str,
        entity_label: str,
        start_offset: int,
        end_offset: int,
        annotator_id: str,
        timestamp: str = None
    ) -> Dict:
        """
        Create a single annotation entry.
        
        Args:
            record_id: Unique identifier for the clinical record
            record_text: Full text of the clinical record
            entity_text: The actual text being annotated
            entity_label: Type of entity (PERSON, DATE, etc.)
            start_offset: Character start position in record_text
            end_offset: Character end position in record_text
            annotator_id: Identifier for the annotator
            timestamp: Annotation timestamp (ISO format)
            
        Returns:
            Annotation dictionary
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        # Validate offsets
        if start_offset < 0 or end_offset > len(record_text) or start_offset >= end_offset:
            raise ValueError(f"Invalid offsets: start={start_offset}, end={end_offset}, text_length={len(record_text)}")
        
        # Validate entity text matches record text at offsets
        extracted_text = record_text[start_offset:end_offset]
        if extracted_text != entity_text:
            raise ValueError(f"Entity text mismatch: expected '{entity_text}', found '{extracted_text}'")
        
        # Validate entity label
        if entity_label not in self.ENTITY_LABELS:
            raise ValueError(f"Unknown entity label: {entity_label}. Valid labels: {list(self.ENTITY_LABELS.keys())}")
        
        annotation = {
            "record_id": record_id,
            "record_text": record_text,
            "entity_text": entity_text,
            "entity_label": entity_label,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "annotator_id": annotator_id,
            "timestamp": timestamp,
            "status": "pending_review",
            "reviewed_by": None,
            "review_timestamp": None
        }
        
        return annotation
    
    def save_annotation(self, annotation: Dict) -> None:
        """
        Save a single annotation to JSONL file.
        
        Args:
            annotation: Annotation dictionary
        """
        self._remove_record_review(annotation['record_id'])
        with open(self.annotations_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(annotation) + '\n')

    def load_record_reviews(self) -> List[Dict]:
        """Load records explicitly marked as fully reviewed by an annotator."""
        if not self.record_reviews_file.exists():
            return []

        reviews = []
        with open(self.record_reviews_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))
        return reviews

    def complete_record_review(
        self,
        record_id: str,
        record_text: str,
        annotator_id: str,
        timestamp: str = None
    ) -> None:
        """Mark a record as fully reviewed, including records with no PII."""
        if not record_id:
            raise ValueError("Record ID cannot be empty")
        if not annotator_id.strip():
            raise ValueError("Annotator ID cannot be empty")
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        review = {
            "record_id": record_id,
            "record_text": record_text,
            "annotator_id": annotator_id.strip(),
            "reviewed_at": timestamp
        }
        reviews = [
            saved for saved in self.load_record_reviews()
            if saved['record_id'] != record_id
        ]
        reviews.append(review)

        with open(self.record_reviews_file, 'w', encoding='utf-8') as f:
            for saved in reviews:
                f.write(json.dumps(saved) + '\n')

    def _remove_record_review(self, record_id: str) -> None:
        """Invalidate a completed review if its entity annotations change."""
        reviews = self.load_record_reviews()
        remaining = [review for review in reviews if review['record_id'] != record_id]
        if len(remaining) == len(reviews):
            return

        with open(self.record_reviews_file, 'w', encoding='utf-8') as f:
            for review in remaining:
                f.write(json.dumps(review) + '\n')
    
    def load_annotations(self) -> List[Dict]:
        """
        Load all annotations from JSONL file.
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        if not self.annotations_file.exists():
            return annotations
        
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
        
        return annotations
    
    def get_annotations_by_record(self, record_id: str) -> List[Dict]:
        """
        Get all annotations for a specific record.
        
        Args:
            record_id: Record identifier
            
        Returns:
            List of annotations for this record
        """
        all_annotations = self.load_annotations()
        return [a for a in all_annotations if a['record_id'] == record_id]
    
    def get_annotation_status(self) -> Dict:
        """
        Get overall annotation status.
        
        Returns:
            Dictionary with status information
        """
        annotations = self.load_annotations()
        
        total_records = len(set(a['record_id'] for a in annotations))
        total_annotations = len(annotations)
        
        reviewed = sum(1 for a in annotations if a['status'] == 'reviewed')
        pending = total_annotations - reviewed
        
        return {
            "total_records": total_records,
            "total_annotations": total_annotations,
            "reviewed": reviewed,
            "pending": pending,
            "completion_rate": reviewed / total_annotations if total_annotations > 0 else 0.0
        }
    
    def mark_reviewed(
        self,
        record_id: str,
        reviewer_id: str,
        timestamp: str = None
    ) -> None:
        """
        Mark all annotations for a record as reviewed.
        
        Args:
            record_id: Record identifier
            reviewer_id: Reviewer identifier
            timestamp: Review timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        annotations = self.load_annotations()
        updated_annotations = []
        
        for annotation in annotations:
            if annotation['record_id'] == record_id:
                annotation['status'] = 'reviewed'
                annotation['reviewed_by'] = reviewer_id
                annotation['review_timestamp'] = timestamp
            updated_annotations.append(annotation)
        
        # Rewrite file with updated annotations
        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            for annotation in updated_annotations:
                f.write(json.dumps(annotation) + '\n')
    
    def delete_annotation(
        self,
        record_id: str,
        entity_text: str,
        start_offset: int
    ) -> bool:
        """
        Delete a specific annotation.
        
        Args:
            record_id: Record identifier
            entity_text: Entity text
            start_offset: Start offset
            
        Returns:
            True if deleted, False if not found
        """
        annotations = self.load_annotations()
        filtered_annotations = []
        
        for annotation in annotations:
            if (annotation['record_id'] == record_id and 
                annotation['entity_text'] == entity_text and 
                annotation['start_offset'] == start_offset):
                continue  # Skip this annotation (delete it)
            filtered_annotations.append(annotation)
        
        if len(filtered_annotations) == len(annotations):
            return False  # Nothing deleted
        
        # Rewrite file
        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            for annotation in filtered_annotations:
                f.write(json.dumps(annotation) + '\n')
        
        self._remove_record_review(record_id)
        return True
    
    def export_annotations(self, output_path: Path = None) -> Path:
        """
        Export annotations to a JSON file.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.annotations_dir / f"annotations_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        annotations = self.load_annotations()
        export_data = {
            "metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_annotations": len(annotations),
                "entity_labels": list(self.ENTITY_LABELS.keys())
            },
            "annotations": annotations,
            "completed_record_reviews": self.load_record_reviews()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path
    
    def get_record_sample(self, df, n: int = 10) -> List[Tuple[str, str]]:
        """
        Get a sample of records for annotation.
        
        Args:
            df: DataFrame with clinical records
            n: Number of records to sample
            
        Returns:
            List of (record_id, record_text) tuples
        """
        from data_pipeline import linearise_dataframe
        
        texts = linearise_dataframe(df)
        sample = texts[:n] if len(texts) >= n else texts
        
        return [(f"record_{i}", text) for i, text in enumerate(sample)]
    
    @staticmethod
    def parse_entities_from_annotations(annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Parse entities from annotation data (static method for NER evaluator).
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Dictionary mapping record_id to list of entity dicts
        """
        entities_by_record = {}
        
        for annotation in annotations:
            record_id = annotation['record_id']
            entity = {
                'text': annotation['entity_text'],
                'label': annotation['entity_label'],
                'start': annotation['start_offset'],
                'end': annotation['end_offset']
            }
            
            if record_id not in entities_by_record:
                entities_by_record[record_id] = []
            
            entities_by_record[record_id].append(entity)
        
        return entities_by_record
