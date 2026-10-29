"""
multi_dataset_evaluator.py
==========================
Multi-dataset evaluation system for ANC, PNC, and MOH 204A registers.
Supports individual and collective evaluation across multiple health registers.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from expert_template_manager import ExpertTemplateManager
from ner_evaluation import NEREvaluator, Entity
from annotation_manager import AnnotationManager

@dataclass
class DatasetMetrics:
    """Metrics for a single dataset."""
    dataset_name: str
    dataset_type: str  # ANC, PNC, UNDER5, MATERNITY
    total_records: int
    records_processed: int
    extraction_rate: float
    facts_extracted: int
    ner_precision: Optional[float] = None
    ner_recall: Optional[float] = None
    ner_f1: Optional[float] = None
    kg_triples_generated: int = 0
    quality_score: float = 0.0

@dataclass
class CollectiveMetrics:
    """Metrics for collective evaluation across datasets."""
    total_datasets: int
    total_records: int
    total_facts_extracted: int
    total_kg_triples: int
    average_ner_precision: float
    average_ner_recall: float
    average_ner_f1: float
    overall_quality_score: float
    dataset_metrics: List[DatasetMetrics]

class MultiDatasetEvaluator:
    """Evaluates performance across multiple health registers."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize multi-dataset evaluator.
        
        Args:
            output_dir: Directory to store evaluation results
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "evaluation_results"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.template_manager = ExpertTemplateManager()
        self.ner_evaluator = NEREvaluator()
        self.annotation_manager = AnnotationManager()
        
        self.evaluation_results_file = self.output_dir / "multi_dataset_evaluation.json"
    
    def evaluate_single_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        dataset_type: str,
        template_id: str = None,
        annotations_dir: Path = None
    ) -> DatasetMetrics:
        """
        Evaluate a single dataset.
        
        Args:
            df: DataFrame with clinical records
            dataset_name: Name of the dataset
            dataset_type: Type of register (ANC, PNC, UNDER5, MATERNITY)
            template_id: Template ID for fact extraction (auto-detected if None)
            annotations_dir: Directory with manual annotations for NER evaluation
            
        Returns:
            DatasetMetrics object
        """
        # Auto-detect template if not provided
        if template_id is None:
            template = self.template_manager.get_template_by_register(dataset_type)
            if template is None:
                raise ValueError(f"No template found for register type: {dataset_type}")
            template_id = template['template_id']
        
        total_records = len(df)
        records_processed = 0
        
        # Extract facts using expert template
        try:
            facts = self.template_manager.extract_facts_from_dataset(df, template_id)
            facts_extracted = len(facts)
            records_processed = total_records
            
            # Convert to KG triples
            template = self.template_manager.load_template(template_id)
            kg_triples = self.template_manager.convert_facts_to_kg_triples(facts, template)
            kg_triples_generated = len(kg_triples)
            
            extraction_rate = facts_extracted / total_records if total_records > 0 else 0.0
        except Exception as e:
            print(f"Error extracting facts from {dataset_name}: {e}")
            facts_extracted = 0
            kg_triples_generated = 0
            extraction_rate = 0.0
        
        # NER evaluation if annotations available
        ner_precision = None
        ner_recall = None
        ner_f1 = None
        
        if annotations_dir and annotations_dir.exists():
            try:
                # Load manual annotations
                annotation_manager = AnnotationManager(annotations_dir)
                annotations = annotation_manager.load_annotations()
                
                if annotations:
                    # Get ground truth entities
                    reference_entities = annotation_manager.parse_entities_from_annotations(annotations)
                    
                    # Generate predictions
                    from data_pipeline import linearise_dataframe, improved_pii_detection_entities
                    texts = linearise_dataframe(df)
                    predicted_entities = {}
                    
                    for i, text in enumerate(texts):
                        record_id = f"{dataset_type}_record_{i}"
                        entities = improved_pii_detection_entities(text)
                        predicted_entities[record_id] = entities
                    
                    # Evaluate
                    results = self.ner_evaluator.evaluate_dataset(
                        predicted_entities,
                        reference_entities
                    )
                    
                    ner_precision = results['overall']['precision']
                    ner_recall = results['overall']['recall']
                    ner_f1 = results['overall']['f1']
            except Exception as e:
                print(f"Error in NER evaluation for {dataset_name}: {e}")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            extraction_rate,
            ner_precision,
            ner_recall,
            kg_triples_generated,
            total_records
        )
        
        return DatasetMetrics(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            total_records=total_records,
            records_processed=records_processed,
            extraction_rate=extraction_rate,
            facts_extracted=facts_extracted,
            ner_precision=ner_precision,
            ner_recall=ner_recall,
            ner_f1=ner_f1,
            kg_triples_generated=kg_triples_generated,
            quality_score=quality_score
        )
    
    def evaluate_collective(
        self,
        datasets: Dict[str, Tuple[pd.DataFrame, str]],
        annotations_dir: Path = None
    ) -> CollectiveMetrics:
        """
        Evaluate multiple datasets collectively.
        
        Args:
            datasets: Dictionary mapping dataset names to (DataFrame, dataset_type) tuples
            annotations_dir: Directory with manual annotations for NER evaluation
            
        Returns:
            CollectiveMetrics object
        """
        dataset_metrics = []
        
        for dataset_name, (df, dataset_type) in datasets.items():
            print(f"Evaluating {dataset_name} ({dataset_type})...")
            
            try:
                metrics = self.evaluate_single_dataset(
                    df,
                    dataset_name,
                    dataset_type,
                    annotations_dir=annotations_dir
                )
                dataset_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                continue
        
        # Calculate collective metrics
        total_datasets = len(dataset_metrics)
        total_records = sum(m.total_records for m in dataset_metrics)
        total_facts_extracted = sum(m.facts_extracted for m in dataset_metrics)
        total_kg_triples = sum(m.kg_triples_generated for m in dataset_metrics)
        
        # Average NER metrics (only for datasets with NER evaluation)
        ner_precisions = [m.ner_precision for m in dataset_metrics if m.ner_precision is not None]
        ner_recalls = [m.ner_recall for m in dataset_metrics if m.ner_recall is not None]
        ner_f1s = [m.ner_f1 for m in dataset_metrics if m.ner_f1 is not None]
        
        average_ner_precision = np.mean(ner_precisions) if ner_precisions else 0.0
        average_ner_recall = np.mean(ner_recalls) if ner_recalls else 0.0
        average_ner_f1 = np.mean(ner_f1s) if ner_f1s else 0.0
        
        # Overall quality score
        overall_quality_score = np.mean([m.quality_score for m in dataset_metrics]) if dataset_metrics else 0.0
        
        return CollectiveMetrics(
            total_datasets=total_datasets,
            total_records=total_records,
            total_facts_extracted=total_facts_extracted,
            total_kg_triples=total_kg_triples,
            average_ner_precision=average_ner_precision,
            average_ner_recall=average_ner_recall,
            average_ner_f1=average_ner_f1,
            overall_quality_score=overall_quality_score,
            dataset_metrics=dataset_metrics
        )
    
    def _calculate_quality_score(
        self,
        extraction_rate: float,
        ner_precision: Optional[float],
        ner_recall: Optional[float],
        kg_triples: int,
        total_records: int
    ) -> float:
        """
        Calculate overall quality score for a dataset.
        
        Args:
            extraction_rate: Rate of successful fact extraction
            ner_precision: NER precision (if available)
            ner_recall: NER recall (if available)
            kg_triples: Number of KG triples generated
            total_records: Total number of records
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Extraction rate contribution (30%)
        score += extraction_rate * 0.3
        
        # NER metrics contribution (40%)
        if ner_precision is not None and ner_recall is not None:
            ner_score = (ner_precision + ner_recall) / 2
            score += ner_score * 0.4
        else:
            # If no NER evaluation, assume baseline 0.7
            score += 0.7 * 0.4
        
        # KG triples contribution (30%)
        triples_per_record = kg_triples / total_records if total_records > 0 else 0
        kg_score = min(triples_per_record / 10, 1.0)  # Cap at 10 triples per record
        score += kg_score * 0.3
        
        return min(score, 1.0)
    
    def save_evaluation_results(
        self,
        metrics: CollectiveMetrics,
        output_path: Path = None
    ) -> Path:
        """
        Save evaluation results to JSON file.
        
        Args:
            metrics: CollectiveMetrics object
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.evaluation_results_file
        
        results = {
            'evaluation_timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_datasets': metrics.total_datasets,
                'total_records': metrics.total_records,
                'total_facts_extracted': metrics.total_facts_extracted,
                'total_kg_triples': metrics.total_kg_triples,
                'average_ner_precision': metrics.average_ner_precision,
                'average_ner_recall': metrics.average_ner_recall,
                'average_ner_f1': metrics.average_ner_f1,
                'overall_quality_score': metrics.overall_quality_score
            },
            'dataset_metrics': [asdict(m) for m in metrics.dataset_metrics]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def load_evaluation_results(self, file_path: Path = None) -> Dict:
        """
        Load evaluation results from JSON file.
        
        Args:
            file_path: Path to results file
            
        Returns:
            Dictionary with evaluation results
        """
        if file_path is None:
            file_path = self.evaluation_results_file
        
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_evaluation_report(self, metrics: CollectiveMetrics) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            metrics: CollectiveMetrics object
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("MULTI-DATASET EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Evaluation Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("")
        
        # Summary
        report.append("COLLECTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Datasets Evaluated: {metrics.total_datasets}")
        report.append(f"Total Records: {metrics.total_records:,}")
        report.append(f"Total Facts Extracted: {metrics.total_facts_extracted:,}")
        report.append(f"Total KG Triples: {metrics.total_kg_triples:,}")
        report.append(f"Average NER Precision: {metrics.average_ner_precision:.2%}" if metrics.average_ner_precision > 0 else "Average NER Precision: N/A")
        report.append(f"Average NER Recall: {metrics.average_ner_recall:.2%}" if metrics.average_ner_recall > 0 else "Average NER Recall: N/A")
        report.append(f"Average NER F1: {metrics.average_ner_f1:.2%}" if metrics.average_ner_f1 > 0 else "Average NER F1: N/A")
        report.append(f"Overall Quality Score: {metrics.overall_quality_score:.2%}")
        report.append("")
        
        # Per-dataset metrics
        report.append("PER-DATASET METRICS")
        report.append("-" * 80)
        
        for dm in metrics.dataset_metrics:
            report.append(f"\nDataset: {dm.dataset_name} ({dm.dataset_type})")
            report.append(f"  Total Records: {dm.total_records:,}")
            report.append(f"  Records Processed: {dm.records_processed:,}")
            report.append(f"  Extraction Rate: {dm.extraction_rate:.2%}")
            report.append(f"  Facts Extracted: {dm.facts_extracted:,}")
            report.append(f"  KG Triples: {dm.kg_triples_generated:,}")
            
            if dm.ner_precision is not None:
                report.append(f"  NER Precision: {dm.ner_precision:.2%}")
                report.append(f"  NER Recall: {dm.ner_recall:.2%}")
                report.append(f"  NER F1: {dm.ner_f1:.2%}")
            else:
                report.append(f"  NER Metrics: Not evaluated (no annotations)")
            
            report.append(f"  Quality Score: {dm.quality_score:.2%}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def compare_datasets(
        self,
        metrics: CollectiveMetrics
    ) -> Dict[str, Any]:
        """
        Compare performance across datasets.
        
        Args:
            metrics: CollectiveMetrics object
            
        Returns:
            Dictionary with comparison results
        """
        if len(metrics.dataset_metrics) < 2:
            return {'error': 'Need at least 2 datasets for comparison'}
        
        comparison = {
            'best_extraction_rate': max(metrics.dataset_metrics, key=lambda m: m.extraction_rate),
            'best_ner_precision': max([m for m in metrics.dataset_metrics if m.ner_precision is not None], key=lambda m: m.ner_precision, default=None),
            'best_ner_recall': max([m for m in metrics.dataset_metrics if m.ner_recall is not None], key=lambda m: m.ner_recall, default=None),
            'best_quality_score': max(metrics.dataset_metrics, key=lambda m: m.quality_score),
            'most_facts_extracted': max(metrics.dataset_metrics, key=lambda m: m.facts_extracted),
            'most_kg_triples': max(metrics.dataset_metrics, key=lambda m: m.kg_triples_generated)
        }
        
        return comparison
