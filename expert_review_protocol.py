"""
expert_review_protocol.py
=========================
Multi-expert review protocol for annotation quality assurance.
Implements inter-annotator agreement and consensus mechanisms.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np

@dataclass
class Expert:
    """Represents an expert annotator."""
    expert_id: str
    name: str
    credentials: str
    specialty: str

@dataclass
class Annotation:
    """Represents a single annotation."""
    record_id: str
    entity_text: str
    entity_label: str
    start_offset: int
    end_offset: int
    annotator_id: str
    timestamp: str

class ExpertReviewProtocol:
    """Manages multi-expert review and consensus process."""
    
    def __init__(self, annotations_dir: Path = None):
        """
        Initialize expert review protocol.
        
        Args:
            annotations_dir: Directory for annotation storage
        """
        if annotations_dir is None:
            annotations_dir = Path(__file__).parent / "expert_reviews"
        
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(exist_ok=True)
        
        self.experts_file = self.annotations_dir / "experts.json"
        self.consensus_file = self.annotations_dir / "consensus_annotations.jsonl"
        self.agreement_file = self.annotations_dir / "inter_annotator_agreement.json"
    
    def register_expert(
        self,
        expert_id: str,
        name: str,
        credentials: str,
        specialty: str
    ) -> None:
        """
        Register an expert annotator.
        
        Args:
            expert_id: Unique expert identifier
            name: Expert's full name
            credentials: Professional credentials
            specialty: Area of specialty
        """
        experts = self._load_experts()
        
        expert = {
            "expert_id": expert_id,
            "name": name,
            "credentials": credentials,
            "specialty": specialty,
            "registered_at": "2026-10-29T00:00:00Z"
        }
        
        experts[expert_id] = expert
        
        self._save_experts(experts)
        print(f"✓ Registered expert: {name} ({expert_id})")
    
    def _load_experts(self) -> Dict:
        """Load registered experts."""
        if not self.experts_file.exists():
            return {}
        
        with open(self.experts_file, 'r') as f:
            return json.load(f)
    
    def _save_experts(self, experts: Dict) -> None:
        """Save experts to file."""
        with open(self.experts_file, 'w') as f:
            json.dump(experts, f, indent=2)
    
    def calculate_inter_annotator_agreement(
        self,
        annotations_by_expert: Dict[str, List[Annotation]]
    ) -> Dict:
        """
        Calculate inter-annotator agreement (Cohen's Kappa).
        
        Args:
            annotations_by_expert: Dictionary mapping expert IDs to their annotations
            
        Returns:
            Agreement metrics
        """
        expert_ids = list(annotations_by_expert.keys())
        
        if len(expert_ids) < 2:
            return {"error": "Need at least 2 experts for agreement calculation"}
        
        # Calculate pairwise agreement
        pairwise_agreements = []
        
        for i in range(len(expert_ids)):
            for j in range(i + 1, len(expert_ids)):
                expert1 = expert_ids[i]
                expert2 = expert_ids[j]
                
                ann1 = annotations_by_expert[expert1]
                ann2 = annotations_by_expert[expert2]
                
                agreement = self._calculate_pairwise_agreement(ann1, ann2)
                pairwise_agreements.append({
                    "expert1": expert1,
                    "expert2": expert2,
                    "agreement": agreement
                })
        
        # Calculate average agreement
        avg_agreement = np.mean([a["agreement"] for a in pairwise_agreements])
        
        results = {
            "pairwise_agreements": pairwise_agreements,
            "average_agreement": avg_agreement,
            "num_experts": len(expert_ids),
            "interpretation": self._interpret_agreement(avg_agreement)
        }
        
        return results
    
    def _calculate_pairwise_agreement(
        self,
        annotations1: List[Annotation],
        annotations2: List[Annotation]
    ) -> float:
        """
        Calculate agreement between two annotators.
        
        Args:
            annotations1: Annotations from expert 1
            annotations2: Annotations from expert 2
            
        Returns:
            Agreement score (0-1)
        """
        # Group by record
        ann1_by_record = self._group_by_record(annotations1)
        ann2_by_record = self._group_by_record(annotations2)
        
        # Get common records
        common_records = set(ann1_by_record.keys()) & set(ann2_by_record.keys())
        
        if not common_records:
            return 0.0
        
        total_entities = 0
        agreements = 0
        
        for record_id in common_records:
            entities1 = ann1_by_record[record_id]
            entities2 = ann2_by_record[record_id]
            
            # Convert to sets for comparison
            set1 = {(e["entity_text"], e["entity_label"], e["start"], e["end"]) for e in entities1}
            set2 = {(e["entity_text"], e["entity_label"], e["start"], e["end"]) for e in entities2}
            
            total_entities += len(set1) + len(set2)
            agreements += len(set1 & set2) * 2  # Count matches twice
        
        if total_entities == 0:
            return 0.0
        
        return agreements / total_entities
    
    def _group_by_record(self, annotations: List[Annotation]) -> Dict:
        """Group annotations by record ID."""
        grouped = {}
        for ann in annotations:
            if ann["record_id"] not in grouped:
                grouped[ann["record_id"]] = []
            grouped[ann["record_id"]].append(ann)
        return grouped
    
    def _interpret_agreement(self, agreement: float) -> str:
        """Interpret agreement score."""
        if agreement >= 0.9:
            return "Excellent agreement (≥0.9)"
        elif agreement >= 0.8:
            return "Good agreement (0.8-0.9)"
        elif agreement >= 0.7:
            return "Acceptable agreement (0.7-0.8)"
        elif agreement >= 0.6:
            return "Moderate agreement (0.6-0.7)"
        else:
            return "Poor agreement (<0.6) - requires adjudication"
    
    def generate_consensus(
        self,
        annotations_by_expert: Dict[str, List[Annotation]],
        method: str = "majority_vote"
    ) -> List[Annotation]:
        """
        Generate consensus annotations from multiple experts.
        
        Args:
            annotations_by_expert: Dictionary mapping expert IDs to annotations
            method: Consensus method (majority_vote, intersection, union)
            
        Returns:
            Consensus annotations
        """
        if method == "majority_vote":
            return self._majority_vote_consensus(annotations_by_expert)
        elif method == "intersection":
            return self._intersection_consensus(annotations_by_expert)
        elif method == "union":
            return self._union_consensus(annotations_by_expert)
        else:
            return self._majority_vote_consensus(annotations_by_expert)
    
    def _majority_vote_consensus(
        self,
        annotations_by_expert: Dict[str, List[Annotation]]
    ) -> List[Annotation]:
        """Generate consensus using majority voting."""
        # Group annotations by record and entity position
        entity_groups = {}
        
        for expert_id, annotations in annotations_by_expert.items():
            for ann in annotations:
                key = (
                    ann["record_id"],
                    ann["start_offset"],
                    ann["end_offset"]
                )
                
                if key not in entity_groups:
                    entity_groups[key] = []
                
                entity_groups[key].append({
                    "entity_text": ann["entity_text"],
                    "entity_label": ann["entity_label"],
                    "annotator_id": expert_id
                })
        
        # For each entity group, take majority vote
        consensus = []
        
        for key, group in entity_groups.items():
            # Count label votes
            label_counts = Counter(e["entity_label"] for e in group)
            majority_label = label_counts.most_common(1)[0][0]
            
            # Get most common entity text
            text_counts = Counter(e["entity_text"] for e in group if e["entity_label"] == majority_label)
            majority_text = text_counts.most_common(1)[0][0]
            
            # Create consensus annotation
            record_id, start, end = key
            consensus_ann = {
                "record_id": record_id,
                "entity_text": majority_text,
                "entity_label": majority_label,
                "start_offset": start,
                "end_offset": end,
                "annotator_id": "consensus",
                "timestamp": "2026-10-29T00:00:00Z",
                "method": "majority_vote",
                "voters": len(group)
            }
            consensus.append(consensus_ann)
        
        return consensus
    
    def _intersection_consensus(
        self,
        annotations_by_expert: Dict[str, List[Annotation]]
    ) -> List[Annotation]:
        """Generate consensus using intersection (all experts must agree)."""
        # Group annotations by record and entity position
        entity_groups = {}
        
        for expert_id, annotations in annotations_by_expert.items():
            for ann in annotations:
                key = (
                    ann["record_id"],
                    ann["start_offset"],
                    ann["end_offset"],
                    ann["entity_label"]
                )
                
                if key not in entity_groups:
                    entity_groups[key] = []
                
                entity_groups[key].append(ann)
        
        # Keep only entities where all experts agree
        consensus = []
        
        for key, group in entity_groups.items():
            if len(group) == len(annotations_by_expert):
                # All experts agree
                record_id, start, end, label = key
                consensus_ann = {
                    "record_id": record_id,
                    "entity_text": group[0]["entity_text"],
                    "entity_label": label,
                    "start_offset": start,
                    "end_offset": end,
                    "annotator_id": "consensus",
                    "timestamp": "2026-10-29T00:00:00Z",
                    "method": "intersection",
                    "unanimous": True
                }
                consensus.append(consensus_ann)
        
        return consensus
    
    def _union_consensus(
        self,
        annotations_by_expert: Dict[str, List[Annotation]]
    ) -> List[Annotation]:
        """Generate consensus using union (any expert's annotation is kept)."""
        consensus = []
        seen = set()
        
        for expert_id, annotations in annotations_by_expert.items():
            for ann in annotations:
                key = (
                    ann["record_id"],
                    ann["start_offset"],
                    ann["end_offset"]
                )
                
                if key not in seen:
                    consensus_ann = {
                        "record_id": ann["record_id"],
                        "entity_text": ann["entity_text"],
                        "entity_label": ann["entity_label"],
                        "start_offset": ann["start_offset"],
                        "end_offset": ann["end_offset"],
                        "annotator_id": expert_id,
                        "timestamp": ann["timestamp"],
                        "method": "union"
                    }
                    consensus.append(consensus_ann)
                    seen.add(key)
        
        return consensus
    
    def save_consensus(self, consensus: List[Annotation]) -> None:
        """Save consensus annotations to file."""
        with open(self.consensus_file, 'w', encoding='utf-8') as f:
            for ann in consensus:
                f.write(json.dumps(ann) + '\n')
        
        print(f"Saved {len(consensus)} consensus annotations to {self.consensus_file}")
    
    def save_agreement_metrics(self, metrics: Dict) -> None:
        """Save inter-annotator agreement metrics."""
        with open(self.agreement_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved agreement metrics to {self.agreement_file}")

# Convenience function
def setup_multi_expert_review(
    num_experts: int = 3
) -> ExpertReviewProtocol:
    """
    Setup multi-expert review protocol.
    
    Args:
        num_experts: Number of experts to register
        
    Returns:
        ExpertReviewProtocol instance
    """
    protocol = ExpertReviewProtocol()
    
    # Register sample experts
    specialties = ["Clinical Medicine", "Health Informatics", "Obstetrics"]
    
    for i in range(num_experts):
        protocol.register_expert(
            expert_id=f"expert_{i+1}",
            name=f"Expert {i+1}",
            credentials=f"MD, PhD (Specialty: {specialties[i % len(specialties)]})",
            specialty=specialties[i % len(specialties)]
        )
    
    return protocol
