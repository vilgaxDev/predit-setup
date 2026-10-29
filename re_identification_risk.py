"""
re_identification_risk.py
=========================
Re-identification risk assessment for anonymized data.
Tests k-anonymity, l-diversity, t-closeness, and linkage attacks.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ReIdentificationRiskAssessment:
    """Assesses re-identification risk of anonymized data."""
    
    def __init__(self, anonymized_df: pd.DataFrame, original_df: pd.DataFrame = None):
        """
        Initialize risk assessment.
        
        Args:
            anonymized_df: Anonymized DataFrame
            original_df: Original DataFrame (for comparison)
        """
        self.anonymized_df = anonymized_df
        self.original_df = original_df
        self.quasi_identifiers = self._identify_quasi_identifiers()
        self.sensitive_attributes = self._identify_sensitive_attributes()
    
    def _identify_quasi_identifiers(self) -> List[str]:
        """Identify quasi-identifier columns."""
        # Common quasi-identifiers in medical data
        quasi_id_cols = []
        
        for col in self.anonymized_df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['age', 'date', 'location', 'village', 'county', 'year', 'month']):
                quasi_id_cols.append(col)
        
        return quasi_id_cols
    
    def _identify_sensitive_attributes(self) -> List[str]:
        """Identify sensitive attribute columns."""
        sensitive_cols = []
        
        for col in self.anonymized_df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['hiv', 'diagnosis', 'condition', 'disease', 'outcome', 'status']):
                sensitive_cols.append(col)
        
        return sensitive_cols
    
    def calculate_k_anonymity(self, k: int = 5) -> Dict:
        """
        Calculate k-anonymity compliance.
        
        Args:
            k: Minimum k value for k-anonymity
            
        Returns:
            k-anonymity assessment results
        """
        if not self.quasi_identifiers:
            return {"error": "No quasi-identifiers identified"}
        
        # Create equivalence classes based on quasi-identifiers
        equivalence_classes = self.anonymized_df.groupby(self.quasi_identifiers).size()
        
        # Count classes that violate k-anonymity
        violating_classes = equivalence_classes[equivalence_classes < k]
        total_classes = len(equivalence_classes)
        violating_count = len(violating_classes)
        
        # Calculate percentage
        violation_rate = (violating_count / total_classes * 100) if total_classes > 0 else 0
        
        # Minimum class size
        min_class_size = equivalence_classes.min() if len(equivalence_classes) > 0 else 0
        
        return {
            "k_value": k,
            "total_equivalence_classes": total_classes,
            "violating_classes": violating_count,
            "violation_rate": violation_rate,
            "min_class_size": min_class_size,
            "k_anonymity_compliant": violating_count == 0,
            "violating_class_sizes": violating_classes.tolist() if len(violating_classes) > 0 else []
        }
    
    def calculate_l_diversity(self, l: int = 3) -> Dict:
        """
        Calculate l-diversity compliance.
        
        Args:
            l: Minimum l value for l-diversity
            
        Returns:
            l-diversity assessment results
        """
        if not self.sensitive_attributes:
            return {"error": "No sensitive attributes identified"}
        
        results = {}
        
        for sensitive_attr in self.sensitive_attributes:
            # For each equivalence class, count distinct sensitive values
            diversity_scores = []
            
            for _, group in self.anonymized_df.groupby(self.quasi_identifiers):
                distinct_values = group[sensitive_attr].nunique()
                diversity_scores.append(distinct_values)
            
            # Count classes violating l-diversity
            violating_classes = sum(1 for score in diversity_scores if score < l)
            total_classes = len(diversity_scores)
            
            results[sensitive_attr] = {
                "l_value": l,
                "total_classes": total_classes,
                "violating_classes": violating_classes,
                "violation_rate": (violating_classes / total_classes * 100) if total_classes > 0 else 0,
                "min_diversity": min(diversity_scores) if diversity_scores else 0,
                "l_diversity_compliant": violating_classes == 0
            }
        
        return results
    
    def calculate_t_closeness(self, t: float = 0.25) -> Dict:
        """
        Calculate t-closeness compliance.
        
        Args:
            t: Maximum threshold for t-closeness
            
        Returns:
            t-closeness assessment results
        """
        if not self.sensitive_attributes:
            return {"error": "No sensitive attributes identified"}
        
        results = {}
        
        for sensitive_attr in self.sensitive_attributes:
            # Overall distribution of sensitive attribute
            overall_dist = self.anonymized_df[sensitive_attr].value_counts(normalize=True)
            
            max_distance = 0
            
            # For each equivalence class, calculate distribution distance
            for _, group in self.anonymized_df.groupby(self.quasi_identifiers):
                class_dist = group[sensitive_attr].value_counts(normalize=True)
                
                # Calculate total variation distance
                distance = 0
                for value in overall_dist.index:
                    overall_prob = overall_dist.get(value, 0)
                    class_prob = class_dist.get(value, 0)
                    distance += abs(overall_prob - class_prob)
                
                distance = distance / 2  # Total variation distance
                max_distance = max(max_distance, distance)
            
            results[sensitive_attr] = {
                "t_value": t,
                "max_distance": max_distance,
                "t_closeness_compliant": max_distance <= t
            }
        
        return results
    
    def check_uniqueness(self) -> Dict:
        """
        Check for unique records that could be re-identified.
        
        Returns:
            Uniqueness assessment results
        """
        if not self.quasi_identifiers:
            return {"error": "No quasi-identifiers identified"}
        
        # Count unique combinations of quasi-identifiers
        unique_combinations = self.anonymized_df[self.quasi_identifiers].drop_duplicates()
        total_records = len(self.anonymized_df)
        unique_count = len(unique_combinations)
        
        # Identify unique records (appear only once)
        record_counts = self.anonymized_df.groupby(self.quasi_identifiers).size()
        unique_records = record_counts[record_counts == 1]
        
        return {
            "total_records": total_records,
            "unique_combinations": unique_count,
            "unique_records_count": len(unique_records),
            "unique_rate": (len(unique_records) / total_records * 100),
            "most_common_combination": record_counts.idxmax() if len(record_counts) > 0 else None,
            "most_common_count": record_counts.max() if len(record_counts) > 0 else 0
        }
    
    def simulate_linkage_attack(self, external_data: pd.DataFrame = None) -> Dict:
        """
        Simulate a linkage attack to test re-identification.
        
        Args:
            external_data: External dataset to link against
            
        Returns:
            Linkage attack results
        """
        if external_data is None:
            # Simulate with subset of original data
            if self.original_df is not None:
                external_data = self.original_df.sample(min(100, len(self.original_df)))
            else:
                return {"error": "No external data provided for linkage attack simulation"}
        
        # Try to match records based on quasi-identifiers
        matches = 0
        total_attempts = len(external_data)
        
        for _, external_record in external_data.iterrows():
            # Find matching quasi-identifiers in anonymized data
            match_found = False
            
            for _, anonymized_record in self.anonymized_df.iterrows():
                if self._match_quasi_identifiers(external_record, anonymized_record):
                    matches += 1
                    match_found = True
                    break
            
            if not match_found:
                pass  # No match found (good for privacy)
        
        linkage_rate = (matches / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_attempts": total_attempts,
            "successful_linkages": matches,
            "linkage_rate": linkage_rate,
            "linkage_risk": "HIGH" if linkage_rate > 50 else "MEDIUM" if linkage_rate > 20 else "LOW"
        }
    
    def _match_quasi_identifiers(self, record1: pd.Series, record2: pd.Series) -> bool:
        """Check if two records match on quasi-identifiers."""
        for col in self.quasi_identifiers:
            if col in record1 and col in record2:
                if str(record1[col]) != str(record2[col]):
                    return False
        return True
    
    def assess_attribute_disclosure(self) -> Dict:
        """
        Assess attribute disclosure risk.
        
        Returns:
            Attribute disclosure assessment
        """
        disclosure_risks = {}
        
        for col in self.anonymized_df.columns:
            # Check if column exists in both dataframes
            if self.original_df is not None and col in self.original_df.columns:
                original_values = self.original_df[col]
                anonymized_values = self.anonymized_df[col]
                
                # Calculate how many unique values remain
                original_unique = original_values.nunique()
                anonymized_unique = anonymized_values.nunique()
                
                # Check for direct value preservation
                preserved = set(original_values) & set(anonymized_values)
                preservation_rate = (len(preserved) / original_unique * 100) if original_unique > 0 else 0
                
                disclosure_risks[col] = {
                    "original_unique": original_unique,
                    "anonymized_unique": anonymized_unique,
                    "preserved_values": len(preserved),
                    "preservation_rate": preservation_rate,
                    "disclosure_risk": "HIGH" if preservation_rate > 50 else "MEDIUM" if preservation_rate > 20 else "LOW"
                }
            else:
                # Column only in anonymized data
                disclosure_risks[col] = {
                    "note": "Column only in anonymized data",
                    "original_unique": "N/A",
                    "anonymized_unique": self.anonymized_df[col].nunique(),
                    "disclosure_risk": "N/A"
                }
        
        return disclosure_risks
    
    def generate_risk_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive risk assessment report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Comprehensive risk report
        """
        report = {
            "dataset_size": int(len(self.anonymized_df)),
            "quasi_identifiers": self.quasi_identifiers,
            "sensitive_attributes": self.sensitive_attributes,
            "k_anonymity": self._convert_to_serializable(self.calculate_k_anonymity(k=5)),
            "l_diversity": self._convert_to_serializable(self.calculate_l_diversity(l=3)),
            "t_closeness": self._convert_to_serializable(self.calculate_t_closeness(t=0.25)),
            "uniqueness": self._convert_to_serializable(self.check_uniqueness()),
            "attribute_disclosure": self._convert_to_serializable(self.assess_attribute_disclosure())
        }
        
        # Overall risk assessment
        risk_score = 0
        risk_factors = []
        
        if not report["k_anonymity"].get("k_anonymity_compliant", True):
            risk_score += 25
            risk_factors.append("k-anonymity violation")
        
        if not report["uniqueness"].get("unique_rate", 0) < 10:
            risk_score += 25
            risk_factors.append("high uniqueness rate")
        
        report["overall_risk_score"] = risk_score
        report["risk_level"] = "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW"
        report["risk_factors"] = risk_factors
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Risk report saved to: {output_path}")
        
        return report
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        else:
            return obj
    
    def visualize_risk_assessment(self, output_dir: str = None) -> None:
        """
        Visualize risk assessment results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "risk_visualizations"
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # k-anonymity visualization
        k_results = self.calculate_k_anonymity(k=5)
        if "error" not in k_results:
            plt.figure(figsize=(10, 6))
            plt.bar(['Compliant', 'Violating'], 
                    [k_results['total_equivalence_classes'] - k_results['violating_classes'],
                     k_results['violating_classes']])
            plt.title(f'k-Anonymity Compliance (k={k_results["k_value"]})')
            plt.ylabel('Number of Equivalence Classes')
            plt.savefig(f"{output_dir}/k_anonymity.png")
            plt.close()
        
        # Uniqueness visualization
        uniqueness = self.check_uniqueness()
        if "error" not in uniqueness:
            plt.figure(figsize=(10, 6))
            plt.pie([uniqueness['unique_records_count'], 
                    uniqueness['total_records'] - uniqueness['unique_records_count']],
                   labels=['Unique Records', 'Non-Unique Records'],
                   autopct='%1.1f%%')
            plt.title('Record Uniqueness')
            plt.savefig(f"{output_dir}/uniqueness.png")
            plt.close()
        
        print(f"Visualizations saved to: {output_dir}")

# Convenience function
def assess_re_identification_risk(
    anonymized_df: pd.DataFrame,
    original_df: pd.DataFrame = None,
    output_dir: str = None
) -> Dict:
    """
    Convenience function to assess re-identification risk.
    
    Args:
        anonymized_df: Anonymized DataFrame
        original_df: Original DataFrame
        output_dir: Output directory for report and visualizations
        
    Returns:
        Risk assessment report
    """
    assessor = ReIdentificationRiskAssessment(anonymized_df, original_df)
    
    report = assessor.generate_risk_report()
    assessor.visualize_risk_assessment(output_dir)
    
    return report
