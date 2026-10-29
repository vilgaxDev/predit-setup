"""
sample_annotations_generator.py
==============================
Generate sample ground truth annotations for demonstration.
Creates annotated records with PII labels for training and evaluation.
"""

import pandas as pd
import random
from pathlib import Path
from typing import List, Dict
import json

class SampleAnnotationGenerator:
    """Generates sample ground truth annotations for training."""
    
    def __init__(self):
        """Initialize annotation generator."""
        self.entity_types = ["PERSON", "DATE", "PATIENT_ID", "LOCATION", "PHONE", "OTHER_PII"]
    
    def generate_sample_annotations(
        self,
        df: pd.DataFrame,
        num_samples: int = 50
    ) -> List[Dict]:
        """
        Generate sample annotations from dataset.
        
        Args:
            df: DataFrame with clinical records
            num_samples: Number of records to annotate
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # Sample records
        sample_indices = random.sample(range(len(df)), min(num_samples, len(df)))
        
        for idx in sample_indices:
            row = df.iloc[idx]
            record_text = self._linearise_record(row)
            record_id = f"record_{idx}"
            
            # Extract entities based on known fields
            entities = self._extract_entities(row, record_text)
            
            for entity in entities:
                annotation = {
                    "record_id": record_id,
                    "record_text": record_text,
                    "entity_text": entity["text"],
                    "entity_label": entity["label"],
                    "start_offset": entity["start"],
                    "end_offset": entity["end"],
                    "annotator_id": "expert_1",
                    "timestamp": "2026-10-29T00:00:00Z",
                    "status": "reviewed",
                    "reviewed_by": "expert_1",
                    "review_timestamp": "2026-10-29T00:00:00Z"
                }
                annotations.append(annotation)
        
        return annotations
    
    def _linearise_record(self, row: pd.Series) -> str:
        """Linearise a record to text format."""
        return " | ".join([f"{col}: {row[col]}" for col in row.index])
    
    def _extract_entities(self, row: pd.Series, text: str) -> List[Dict]:
        """Extract entities from a record based on known fields."""
        entities = []
        current_pos = 0
        
        # Helper to find field position
        def find_field_position(field_name: str) -> int:
            pattern = f"{field_name}:"
            pos = text.find(pattern)
            return pos if pos != -1 else -1
        
        # Extract Name
        name_pos = find_field_position("Name")
        if name_pos != -1:
            name_value = row.get("Name", "")
            if pd.notna(name_value) and name_value:
                value_pos = text.find(str(name_value), name_pos)
                if value_pos != -1:
                    entities.append({
                        "text": str(name_value),
                        "label": "PERSON",
                        "start": value_pos,
                        "end": value_pos + len(str(name_value))
                    })
        
        # Extract Village (Location)
        village_pos = find_field_position("Village")
        if village_pos != -1:
            village_value = row.get("Village", "")
            if pd.notna(village_value) and village_value:
                value_pos = text.find(str(village_value), village_pos)
                if value_pos != -1:
                    entities.append({
                        "text": str(village_value),
                        "label": "LOCATION",
                        "start": value_pos,
                        "end": value_pos + len(str(village_value))
                    })
        
        # Extract Date (Date_of_Admission)
        date_pos = find_field_position("Date_of_Admission")
        if date_pos != -1:
            date_value = row.get("Date_of_Admission", "")
            if pd.notna(date_value) and date_value:
                value_pos = text.find(str(date_value), date_pos)
                if value_pos != -1:
                    entities.append({
                        "text": str(date_value),
                        "label": "DATE",
                        "start": value_pos,
                        "end": value_pos + len(str(date_value))
                    })
        
        # Extract Admission_Number (PATIENT_ID)
        id_pos = find_field_position("Admission_Number")
        if id_pos != -1:
            id_value = row.get("Admission_Number", "")
            if pd.notna(id_value) and id_value:
                value_pos = text.find(str(id_value), id_pos)
                if value_pos != -1:
                    entities.append({
                        "text": str(id_value),
                        "label": "PATIENT_ID",
                        "start": value_pos,
                        "end": value_pos + len(str(id_value))
                    })
        
        return entities
    
    def save_annotations(
        self,
        annotations: List[Dict],
        output_path: str = None
    ) -> str:
        """
        Save annotations to JSONL file.
        
        Args:
            annotations: List of annotation dictionaries
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = Path(__file__).parent / "annotations" / "sample_annotations.jsonl"
        
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + '\n')
        
        print(f"Saved {len(annotations)} annotations to {output_path}")
        return output_path
    
    def create_record_reviews(
        self,
        annotations: List[Dict],
        output_path: str = None
    ) -> str:
        """
        Create record reviews for fully reviewed records.
        
        Args:
            annotations: List of annotation dictionaries
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = Path(__file__).parent / "annotations" / "sample_record_reviews.jsonl"
        
        Path(output_path).parent.mkdir(exist_ok=True)
        
        # Get unique record IDs
        record_ids = set(a["record_id"] for a in annotations)
        
        reviews = []
        for record_id in record_ids:
            # Get first annotation for this record to get text
            record_text = next(a["record_text"] for a in annotations if a["record_id"] == record_id)
            
            review = {
                "record_id": record_id,
                "record_text": record_text,
                "annotator_id": "expert_1",
                "reviewed_at": "2026-10-29T00:00:00Z"
            }
            reviews.append(review)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for review in reviews:
                f.write(json.dumps(review) + '\n')
        
        print(f"Created {len(reviews)} record reviews at {output_path}")
        return output_path

# Convenience function
def generate_sample_ground_truth(
    df: pd.DataFrame,
    num_samples: int = 50
) -> str:
    """
    Generate sample ground truth annotations.
    
    Args:
        df: DataFrame with clinical records
        num_samples: Number of records to annotate
        
    Returns:
        Path to annotations file
    """
    generator = SampleAnnotationGenerator()
    annotations = generator.generate_sample_annotations(df, num_samples)
    generator.save_annotations(annotations)
    generator.create_record_reviews(annotations)
    
    return Path(__file__).parent / "annotations" / "sample_annotations.jsonl"
