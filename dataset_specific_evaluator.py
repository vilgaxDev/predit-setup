"""
dataset_specific_evaluator.py
=============================
Dataset-specific evaluation for ANC, PNC, and MOH 204A registers.
Implements validation rules and quality metrics specific to each register type.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class ANCEvaluationResult:
    """Evaluation results for ANC register."""
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    validation_rate: float
    
    # ANC-specific metrics
    average_anc_visits: float
    hiv_positive_rate: float
    hypertension_detection_rate: float
    anemia_detection_rate: float
    bp_measurement_completeness: float
    hemoglobin_completeness: float
    
    # Quality indicators
    first_trimester_booking_rate: float
    tetanus_coverage: float
    ipt_coverage: float
    
    validation_errors: List[str]

@dataclass
class PNCEvaluationResult:
    """Evaluation results for PNC register."""
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    validation_rate: float
    
    # PNC-specific metrics
    maternal_survival_rate: float
    neonatal_survival_rate: float
    early_pnc_visit_rate: float
    breastfeeding_initiation_rate: float
    postpartum_hemorrhage_rate: float
    
    # Quality indicators
    pnc_visit_completeness: float
    follow_up_adherence: float
    
    validation_errors: List[str]

@dataclass
class Under5EvaluationResult:
    """Evaluation results for Under-5 register."""
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    validation_rate: float
    
    # Under-5 specific metrics
    malnutrition_severe_rate: float
    malnutrition_moderate_rate: float
    underweight_rate: float
    stunting_rate: float
    vaccination_complete_rate: float
    
    # Quality indicators
    growth_monitoring_completeness: float
    muac_measurement_completeness: float
    
    validation_errors: List[str]

class DatasetSpecificEvaluator:
    """Evaluates dataset-specific quality metrics for health registers."""
    
    def __init__(self):
        """Initialize dataset-specific evaluator."""
        self.anc_rules = self._get_anc_validation_rules()
        self.pnc_rules = self._get_pnc_validation_rules()
        self.under5_rules = self._get_under5_validation_rules()
    
    def evaluate_anc_register(
        self,
        df: pd.DataFrame,
        dataset_name: str = "ANC Register"
    ) -> ANCEvaluationResult:
        """
        Evaluate ANC register with ANC-specific metrics.
        
        Args:
            df: DataFrame with ANC records
            dataset_name: Name of the dataset
            
        Returns:
            ANCEvaluationResult object
        """
        total_records = len(df)
        validation_errors = []
        valid_records = 0
        
        # Validation checks
        for idx, row in df.iterrows():
            record_errors = self._validate_anc_record(row)
            if not record_errors:
                valid_records += 1
            else:
                validation_errors.extend([f"Record {idx}: {err}" for err in record_errors])
        
        invalid_records = total_records - valid_records
        validation_rate = valid_records / total_records if total_records > 0 else 0.0
        
        # ANC-specific metrics
        average_anc_visits = self._calculate_average_anc_visits(df)
        hiv_positive_rate = self._calculate_hiv_positive_rate(df)
        hypertension_detection_rate = self._calculate_hypertension_rate(df)
        anemia_detection_rate = self._calculate_anemia_rate(df)
        bp_measurement_completeness = self._calculate_bp_completeness(df)
        hemoglobin_completeness = self._calculate_hemoglobin_completeness(df)
        
        # Quality indicators
        first_trimester_booking_rate = self._calculate_first_trimester_booking(df)
        tetanus_coverage = self._calculate_tetanus_coverage(df)
        ipt_coverage = self._calculate_ipt_coverage(df)
        
        return ANCEvaluationResult(
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            validation_rate=validation_rate,
            average_anc_visits=average_anc_visits,
            hiv_positive_rate=hiv_positive_rate,
            hypertension_detection_rate=hypertension_detection_rate,
            anemia_detection_rate=anemia_detection_rate,
            bp_measurement_completeness=bp_measurement_completeness,
            hemoglobin_completeness=hemoglobin_completeness,
            first_trimester_booking_rate=first_trimester_booking_rate,
            tetanus_coverage=tetanus_coverage,
            ipt_coverage=ipt_coverage,
            validation_errors=validation_errors[:50]  # Limit to first 50 errors
        )
    
    def evaluate_pnc_register(
        self,
        df: pd.DataFrame,
        dataset_name: str = "PNC Register"
    ) -> PNCEvaluationResult:
        """
        Evaluate PNC register with PNC-specific metrics.
        
        Args:
            df: DataFrame with PNC records
            dataset_name: Name of the dataset
            
        Returns:
            PNCEvaluationResult object
        """
        total_records = len(df)
        validation_errors = []
        valid_records = 0
        
        # Validation checks
        for idx, row in df.iterrows():
            record_errors = self._validate_pnc_record(row)
            if not record_errors:
                valid_records += 1
            else:
                validation_errors.extend([f"Record {idx}: {err}" for err in record_errors])
        
        invalid_records = total_records - valid_records
        validation_rate = valid_records / total_records if total_records > 0 else 0.0
        
        # PNC-specific metrics
        maternal_survival_rate = self._calculate_maternal_survival_rate(df)
        neonatal_survival_rate = self._calculate_neonatal_survival_rate(df)
        early_pnc_visit_rate = self._calculate_early_pnc_visit_rate(df)
        breastfeeding_initiation_rate = self._calculate_breastfeeding_rate(df)
        postpartum_hemorrhage_rate = self._calculate_pph_rate(df)
        
        # Quality indicators
        pnc_visit_completeness = self._calculate_pnc_completeness(df)
        follow_up_adherence = self._calculate_follow_up_adherence(df)
        
        return PNCEvaluationResult(
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            validation_rate=validation_rate,
            maternal_survival_rate=maternal_survival_rate,
            neonatal_survival_rate=neonatal_survival_rate,
            early_pnc_visit_rate=early_pnc_visit_rate,
            breastfeeding_initiation_rate=breastfeeding_initiation_rate,
            postpartum_hemorrhage_rate=postpartum_hemorrhage_rate,
            pnc_visit_completeness=pnc_visit_completeness,
            follow_up_adherence=follow_up_adherence,
            validation_errors=validation_errors[:50]
        )
    
    def evaluate_under5_register(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Under-5 Register"
    ) -> Under5EvaluationResult:
        """
        Evaluate Under-5 register with Under-5-specific metrics.
        
        Args:
            df: DataFrame with Under-5 records
            dataset_name: Name of the dataset
            
        Returns:
            Under5EvaluationResult object
        """
        total_records = len(df)
        validation_errors = []
        valid_records = 0
        
        # Validation checks
        for idx, row in df.iterrows():
            record_errors = self._validate_under5_record(row)
            if not record_errors:
                valid_records += 1
            else:
                validation_errors.extend([f"Record {idx}: {err}" for err in record_errors])
        
        invalid_records = total_records - valid_records
        validation_rate = valid_records / total_records if total_records > 0 else 0.0
        
        # Under-5 specific metrics
        malnutrition_severe_rate = self._calculate_sam_rate(df)
        malnutrition_moderate_rate = self._calculate_mam_rate(df)
        underweight_rate = self._calculate_underweight_rate(df)
        stunting_rate = self._calculate_stunting_rate(df)
        vaccination_complete_rate = self._calculate_vaccination_rate(df)
        
        # Quality indicators
        growth_monitoring_completeness = self._calculate_growth_monitoring(df)
        muac_measurement_completeness = self._calculate_muac_completeness(df)
        
        return Under5EvaluationResult(
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            validation_rate=validation_rate,
            malnutrition_severe_rate=malnutrition_severe_rate,
            malnutrition_moderate_rate=malnutrition_moderate_rate,
            underweight_rate=underweight_rate,
            stunting_rate=stunting_rate,
            vaccination_complete_rate=vaccination_complete_rate,
            growth_monitoring_completeness=growth_monitoring_completeness,
            muac_measurement_completeness=muac_measurement_completeness,
            validation_errors=validation_errors[:50]
        )
    
    # ANC-specific methods
    def _get_anc_validation_rules(self) -> Dict:
        """Get ANC validation rules."""
        return {
            'gestational_age_range': (0, 42),
            'bp_systolic_range': (60, 240),
            'bp_diastolic_range': (40, 160),
            'hemoglobin_range': (3.0, 20.0),
            'anc_visits_range': (0, 20)
        }
    
    def _validate_anc_record(self, row: pd.Series) -> List[str]:
        """Validate a single ANC record."""
        errors = []
        rules = self.anc_rules
        
        # Check gestational age
        gestation = self._get_numeric_value(row, ['Gestation', 'Gestational Age', 'Gestation at Birth (Weeks)'])
        if gestation is not None:
            if not (rules['gestational_age_range'][0] <= gestation <= rules['gestational_age_range'][1]):
                errors.append(f"Invalid gestational age: {gestation}")
        
        # Check blood pressure
        systolic = self._get_numeric_value(row, ['Systolic BP', 'Blood Pressure Systolic', 'systolic_bp'])
        diastolic = self._get_numeric_value(row, ['Diastolic BP', 'Blood Pressure Diastolic', 'diastolic_bp'])
        
        if systolic is not None and not (rules['bp_systolic_range'][0] <= systolic <= rules['bp_systolic_range'][1]):
            errors.append(f"Invalid systolic BP: {systolic}")
        
        if diastolic is not None and not (rules['bp_diastolic_range'][0] <= diastolic <= rules['bp_diastolic_range'][1]):
            errors.append(f"Invalid diastolic BP: {diastolic}")
        
        if systolic is not None and diastolic is not None and diastolic >= systolic:
            errors.append(f"Diastolic BP ({diastolic}) >= Systolic BP ({systolic})")
        
        # Check hemoglobin
        hemoglobin = self._get_numeric_value(row, ['Hemoglobin', 'Hb', 'haemoglobin'])
        if hemoglobin is not None and not (rules['hemoglobin_range'][0] <= hemoglobin <= rules['hemoglobin_range'][1]):
            errors.append(f"Invalid hemoglobin: {hemoglobin}")
        
        return errors
    
    def _calculate_average_anc_visits(self, df: pd.DataFrame) -> float:
        """Calculate average ANC visits."""
        col = self._find_column(df, ['ANC Visits', 'Total ANC Visits', 'No. of ANC Visits', 'total_anc_visits'])
        if col is None:
            return 0.0
        return df[col].mean() if not df[col].empty else 0.0
    
    def _calculate_hiv_positive_rate(self, df: pd.DataFrame) -> float:
        """Calculate HIV positive rate."""
        col = self._find_column(df, ['HIV Status', 'HIV Status (P/N/KP/U)', 'HIV_Status'])
        if col is None:
            return 0.0
        
        positive = df[df[col].astype(str).str.upper() == 'P'].shape[0]
        total = df[col].notna().sum()
        return positive / total if total > 0 else 0.0
    
    def _calculate_hypertension_rate(self, df: pd.DataFrame) -> float:
        """Calculate hypertension detection rate."""
        systolic_col = self._find_column(df, ['Systolic BP', 'Blood Pressure Systolic', 'systolic_bp'])
        diastolic_col = self._find_column(df, ['Diastolic BP', 'Blood Pressure Diastolic', 'diastolic_bp'])
        
        if systolic_col is None or diastolic_col is None:
            return 0.0
        
        hypertension = df[
            (df[systolic_col] >= 140) | (df[diastolic_col] >= 90)
        ].shape[0]
        total = df[systolic_col].notna().sum()
        return hypertension / total if total > 0 else 0.0
    
    def _calculate_anemia_rate(self, df: pd.DataFrame) -> float:
        """Calculate anemia detection rate."""
        col = self._find_column(df, ['Hemoglobin', 'Hb', 'haemoglobin'])
        if col is None:
            return 0.0
        
        anemia = df[df[col] < 11.0].shape[0]
        total = df[col].notna().sum()
        return anemia / total if total > 0 else 0.0
    
    def _calculate_bp_completeness(self, df: pd.DataFrame) -> float:
        """Calculate BP measurement completeness."""
        systolic_col = self._find_column(df, ['Systolic BP', 'Blood Pressure Systolic', 'systolic_bp'])
        diastolic_col = self._find_column(df, ['Diastolic BP', 'Blood Pressure Diastolic', 'diastolic_bp'])
        
        if systolic_col is None or diastolic_col is None:
            return 0.0
        
        complete = df[systolic_col].notna() & df[diastolic_col].notna()
        return complete.sum() / len(df) if len(df) > 0 else 0.0
    
    def _calculate_hemoglobin_completeness(self, df: pd.DataFrame) -> float:
        """Calculate hemoglobin measurement completeness."""
        col = self._find_column(df, ['Hemoglobin', 'Hb', 'haemoglobin'])
        if col is None:
            return 0.0
        return df[col].notna().sum() / len(df) if len(df) > 0 else 0.0
    
    def _calculate_first_trimester_booking(self, df: pd.DataFrame) -> float:
        """Calculate first trimester booking rate."""
        gestation_col = self._find_column(df, ['Gestation', 'Gestational Age', 'Gestation at Booking'])
        if gestation_col is None:
            return 0.0
        
        first_trimester = df[df[gestation_col] <= 12].shape[0]
        total = df[gestation_col].notna().sum()
        return first_trimester / total if total > 0 else 0.0
    
    def _calculate_tetanus_coverage(self, df: pd.DataFrame) -> float:
        """Calculate tetanus toxoid coverage."""
        col = self._find_column(df, ['Tetanus Toxoid', 'TT', 'tetanus'])
        if col is None:
            return 0.0
        
        vaccinated = df[df[col].astype(str).str.upper() == 'Y'].shape[0]
        total = df[col].notna().sum()
        return vaccinated / total if total > 0 else 0.0
    
    def _calculate_ipt_coverage(self, df: pd.DataFrame) -> float:
        """Calculate IPT coverage."""
        col = self._find_column(df, ['IPT', 'Intermittent Prophylactic Treatment', 'ipt'])
        if col is None:
            return 0.0
        
        ipt = df[df[col].astype(str).str.upper() == 'Y'].shape[0]
        total = df[col].notna().sum()
        return ipt / total if total > 0 else 0.0
    
    # PNC-specific methods
    def _get_pnc_validation_rules(self) -> Dict:
        """Get PNC validation rules."""
        return {
            'postnatal_day_range': (0, 365),
            'survival_codes': ['A', 'D', 'Alive', 'Deceased']
        }
    
    def _validate_pnc_record(self, row: pd.Series) -> List[str]:
        """Validate a single PNC record."""
        errors = []
        rules = self._pnc_rules
        
        # Check postnatal day
        day = self._get_numeric_value(row, ['Postnatal Day', 'Day Postpartum', 'PNC Day'])
        if day is not None and not (rules['postnatal_day_range'][0] <= day <= rules['postnatal_day_range'][1]):
            errors.append(f"Invalid postnatal day: {day}")
        
        return errors
    
    def _calculate_maternal_survival_rate(self, df: pd.DataFrame) -> float:
        """Calculate maternal survival rate."""
        col = self._find_column(df, ['Maternal Survival', "Mother's Status", 'maternal_survival'])
        if col is None:
            return 0.0
        
        alive = df[df[col].astype(str).str.upper() == 'A'].shape[0]
        total = df[col].notna().sum()
        return alive / total if total > 0 else 0.0
    
    def _calculate_neonatal_survival_rate(self, df: pd.DataFrame) -> float:
        """Calculate neonatal survival rate."""
        col = self._find_column(df, ['Child Survival', "Child's Status", 'child_survival'])
        if col is None:
            return 0.0
        
        alive = df[df[col].astype(str).str.upper() == 'A'].shape[0]
        total = df[col].notna().sum()
        return alive / total if total > 0 else 0.0
    
    def _calculate_early_pnc_visit_rate(self, df: pd.DataFrame) -> float:
        """Calculate early PNC visit rate (within 7 days)."""
        day_col = self._find_column(df, ['Postnatal Day', 'Day Postpartum', 'PNC Day'])
        if day_col is None:
            return 0.0
        
        early = df[df[day_col] <= 7].shape[0]
        total = df[day_col].notna().sum()
        return early / total if total > 0 else 0.0
    
    def _calculate_breastfeeding_rate(self, df: pd.DataFrame) -> float:
        """Calculate breastfeeding initiation rate."""
        col = self._find_column(df, ['Breastfeeding', 'BF Initiation', 'breastfeeding'])
        if col is None:
            return 0.0
        
        bf = df[df[col].astype(str).str.upper() == 'Y'].shape[0]
        total = df[col].notna().sum()
        return bf / total if total > 0 else 0.0
    
    def _calculate_pph_rate(self, df: pd.DataFrame) -> float:
        """Calculate postpartum hemorrhage rate."""
        col = self._find_column(df, ['PPH', 'Postpartum Hemorrhage', 'postpartum_hemorrhage'])
        if col is None:
            return 0.0
        
        pph = df[df[col].astype(str).str.upper() == 'Y'].shape[0]
        total = df[col].notna().sum()
        return pph / total if total > 0 else 0.0
    
    def _calculate_pnc_completeness(self, df: pd.DataFrame) -> float:
        """Calculate PNC visit completeness."""
        # Assume 3 recommended PNC visits
        col = self._find_column(df, ['PNC Visits', 'Total PNC Visits', 'pnc_visits'])
        if col is None:
            return 0.0
        
        complete = df[df[col] >= 3].shape[0]
        total = df[col].notna().sum()
        return complete / total if total > 0 else 0.0
    
    def _calculate_follow_up_adherence(self, df: pd.DataFrame) -> float:
        """Calculate follow-up adherence."""
        day_col = self._find_column(df, ['Postnatal Day', 'Day Postpartum', 'PNC Day'])
        if day_col is None:
            return 0.0
        
        # Count visits within recommended timeframes
        total = df[day_col].notna().sum()
        return total / len(df) if len(df) > 0 else 0.0
    
    # Under-5 specific methods
    def _get_under5_validation_rules(self) -> Dict:
        """Get Under-5 validation rules."""
        return {
            'age_months_range': (0, 60),
            'weight_range': (0, 50),
            'height_range': (0, 150),
            'muac_range': (50, 200)
        }
    
    def _validate_under5_record(self, row: pd.Series) -> List[str]:
        """Validate a single Under-5 record."""
        errors = []
        rules = self._under5_rules
        
        # Check age
        age = self._get_numeric_value(row, ['Age Months', 'Age in Months', 'age_months'])
        if age is not None and not (rules['age_months_range'][0] <= age <= rules['age_months_range'][1]):
            errors.append(f"Invalid age: {age}")
        
        # Check weight
        weight = self._get_numeric_value(row, ['Weight', 'Weight kg', 'weight'])
        if weight is not None and not (rules['weight_range'][0] <= weight <= rules['weight_range'][1]):
            errors.append(f"Invalid weight: {weight}")
        
        return errors
    
    def _calculate_sam_rate(self, df: pd.DataFrame) -> float:
        """Calculate severe acute malnutrition rate."""
        muac_col = self._find_column(df, ['MUAC', 'Mid-upper arm circumference', 'muac'])
        if muac_col is None:
            return 0.0
        
        sam = df[df[muac_col] < 115].shape[0]
        total = df[muac_col].notna().sum()
        return sam / total if total > 0 else 0.0
    
    def _calculate_mam_rate(self, df: pd.DataFrame) -> float:
        """Calculate moderate acute malnutrition rate."""
        muac_col = self._find_column(df, ['MUAC', 'Mid-upper arm circumference', 'muac'])
        if muac_col is None:
            return 0.0
        
        mam = df[(df[muac_col] >= 115) & (df[muac_col] < 125)].shape[0]
        total = df[muac_col].notna().sum()
        return mam / total if total > 0 else 0.0
    
    def _calculate_underweight_rate(self, df: pd.DataFrame) -> float:
        """Calculate underweight rate."""
        weight_col = self._find_column(df, ['Weight', 'Weight kg', 'weight'])
        age_col = self._find_column(df, ['Age Months', 'Age in Months', 'age_months'])
        
        if weight_col is None or age_col is None:
            return 0.0
        
        # Simplified: assume underweight if weight < 5kg for any age
        underweight = df[df[weight_col] < 5].shape[0]
        total = df[weight_col].notna().sum()
        return underweight / total if total > 0 else 0.0
    
    def _calculate_stunting_rate(self, df: pd.DataFrame) -> float:
        """Calculate stunting rate."""
        height_col = self._find_column(df, ['Height', 'Height cm', 'height'])
        age_col = self._find_column(df, ['Age Months', 'Age in Months', 'age_months'])
        
        if height_col is None or age_col is None:
            return 0.0
        
        # Simplified: assume stunting if height < 50cm for age > 12 months
        stunting = df[(df[age_col] > 12) & (df[height_col] < 50)].shape[0]
        total = df[height_col].notna().sum()
        return stunting / total if total > 0 else 0.0
    
    def _calculate_vaccination_rate(self, df: pd.DataFrame) -> float:
        """Calculate vaccination completion rate."""
        col = self._find_column(df, ['Vaccination', 'Vaccination Status', 'vaccination'])
        if col is None:
            return 0.0
        
        complete = df[df[col].astype(str).str.upper() == 'C'].shape[0]
        total = df[col].notna().sum()
        return complete / total if total > 0 else 0.0
    
    def _calculate_growth_monitoring(self, df: pd.DataFrame) -> float:
        """Calculate growth monitoring completeness."""
        weight_col = self._find_column(df, ['Weight', 'Weight kg', 'weight'])
        height_col = self._find_column(df, ['Height', 'Height cm', 'height'])
        
        if weight_col is None or height_col is None:
            return 0.0
        
        complete = df[weight_col].notna() & df[height_col].notna()
        return complete.sum() / len(df) if len(df) > 0 else 0.0
    
    def _calculate_muac_completeness(self, df: pd.DataFrame) -> float:
        """Calculate MUAC measurement completeness."""
        col = self._find_column(df, ['MUAC', 'Mid-upper arm circumference', 'muac'])
        if col is None:
            return 0.0
        return df[col].notna().sum() / len(df) if len(df) > 0 else 0.0
    
    # Helper methods
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _get_numeric_value(self, row: pd.Series, possible_names: List[str]) -> Optional[float]:
        """Get numeric value from row by possible column names."""
        for name in possible_names:
            if name in row.index:
                try:
                    return pd.to_numeric(row[name], errors='coerce')
                except:
                    continue
        return None
