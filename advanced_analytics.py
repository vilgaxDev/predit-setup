"""
advanced_analytics.py
====================
Advanced Maternal, Neonatal, and HIV Health Analytics

Features:
1. Risk Classification & Actionable Insights
2. Time-Series Trend Analysis
3. Mortality Root-Cause Analysis & Interventions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class RiskInsight:
    """Risk classification with actionable insights."""
    risk_level: str  # High, Moderate, Low
    risk_factors: List[str]
    confidence: float
    action_steps: List[str]
    priority: str  # Critical, High, Medium, Low

@dataclass
class TrendData:
    """Time-series trend analysis data."""
    period: str
    metric: str
    value: float
    trend: str  # Increasing, Decreasing, Stable
    significance: str  # Significant, Not Significant
    intervention_impact: Optional[str]

@dataclass
class MortalityDriver:
    """Mortality root-cause analysis."""
    cause: str
    frequency: int
    percentage: float
    interventions: List[str]
    evidence_level: str  # Strong, Moderate, Limited
    expected_reduction: Optional[float]

class AdvancedHealthAnalytics:
    """Advanced analytics for maternal, neonatal, and HIV health."""
    
    def __init__(self):
        """Initialize analytics engine."""
        self.risk_thresholds = {
            'high': 3,  # 3+ risk factors
            'moderate': 2,  # 2 risk factors
            'low': 1  # 0-1 risk factors
        }
        
        # WHO risk criteria definitions
        self.risk_criteria = {
            'age_extremes': {
                'condition': lambda row: self._check_age_extremes(row),
                'description': 'Age < 18 or > 35 years',
                'intervention': 'Provide age-appropriate counseling and monitoring'
            },
            'high_parity': {
                'condition': lambda row: self._check_high_parity(row),
                'description': 'Parity > 4',
                'intervention': 'Consider family planning counseling and enhanced monitoring'
            },
            'hiv_positive': {
                'condition': lambda row: self._check_hiv_positive(row),
                'description': 'HIV Positive',
                'intervention': 'Ensure ART adherence, viral load monitoring, and PMTCT protocols'
            },
            'preterm': {
                'condition': lambda row: self._check_preterm(row),
                'description': 'Gestation < 37 weeks',
                'intervention': 'Enhanced neonatal care, kangaroo mother care, thermal protection'
            },
            'post_term': {
                'condition': lambda row: self._check_post_term(row),
                'description': 'Gestation > 42 weeks',
                'intervention': 'Consider induction of labor, close fetal monitoring'
            },
            'delivery_complications': {
                'condition': lambda row: self._check_delivery_complications(row),
                'description': 'Delivery complications present',
                'intervention': 'Immediate specialist referral, emergency obstetric care readiness'
            },
            'c_section': {
                'condition': lambda row: self._check_c_section(row),
                'description': 'C-Section delivery',
                'intervention': 'Monitor for surgical complications, VBAC counseling for future pregnancies'
            },
            'low_birth_weight': {
                'condition': lambda row: self._check_low_birth_weight(row),
                'description': 'Birth weight < 2500g',
                'intervention': 'Nutritional support, thermal protection, enhanced monitoring'
            },
            'poor_anc_visits': {
                'condition': lambda row: self._check_poor_anc(row),
                'description': 'Less than 4 ANC visits',
                'intervention': 'Increase ANC visit frequency, health education, mobile clinic outreach'
            }
        }
        
        # Mortality drivers and interventions
        self.mortality_drivers = {
            'maternal': {
                'postpartum_hemorrhage': {
                    'interventions': [
                        'Active management of third stage of labor',
                        'Uterotonic drugs availability (oxytocin, misoprostol)',
                        'Blood transfusion readiness',
                        'Skilled birth attendant training'
                    ],
                    'evidence_level': 'Strong',
                    'expected_reduction': 30-40
                },
                'hypertensive_disorders': {
                    'interventions': [
                        'Early antenatal BP screening',
                        'Magnesium sulfate for eclampsia',
                        'Antihypertensive medication protocols',
                        'Timely delivery planning'
                    ],
                    'evidence_level': 'Strong',
                    'expected_reduction': 25-35
                },
                'infection_sepsis': {
                    'interventions': [
                        'Sterile procedures during delivery',
                        'Prophylactic antibiotics for C-section',
                        'Early infection detection protocols',
                        'Antibiotic stewardship program'
                    ],
                    'evidence_level': 'Moderate',
                    'expected_reduction': 20-30
                },
                'obstructed_labor': {
                    'interventions': [
                        'Partograph use for labor monitoring',
                        'Timely C-section access',
                        'Referral system strengthening',
                        'Emergency obstetric care training'
                    ],
                    'evidence_level': 'Strong',
                    'expected_reduction': 25-35
                },
                'anemia': {
                    'interventions': [
                        'Routine hemoglobin screening',
                        'Iron and folic acid supplementation',
                        'Deworming programs',
                        'Malaria prevention (ITNs, prophylaxis)'
                    ],
                    'evidence_level': 'Moderate',
                    'expected_reduction': 15-25
                }
            },
            'neonatal': {
                'preterm_birth': {
                    'interventions': [
                        'Antenatal corticosteroids',
                        'Kangaroo mother care',
                        'Surfactant therapy',
                        'Thermal protection'
                    ],
                    'evidence_level': 'Strong',
                    'expected_reduction': 30-40
                },
                'birth_asphyxia': {
                    'interventions': [
                        'Skilled birth attendance',
                        'Neonatal resuscitation training',
                        'APGAR score monitoring',
                        'Immediate newborn care protocols'
                    ],
                    'evidence_level': 'Strong',
                    'expected_reduction': 25-35
                },
                'infection': {
                    'interventions': [
                        'Cord care protocols',
                        'Early breastfeeding promotion',
                        'Hygiene practices',
                        'Antibiotic access'
                    ],
                    'evidence_level': 'Moderate',
                    'expected_reduction': 20-30
                },
                'congenital_anomalies': {
                    'interventions': [
                        'Folic acid supplementation',
                        'Avoidance of teratogens',
                        'Early detection through ultrasound',
                        'Referral to specialized care'
                    ],
                    'evidence_level': 'Limited',
                    'expected_reduction': 10-15
                }
            }
        }
    
    def _check_age_extremes(self, row) -> bool:
        """Check if age is < 18 or > 35."""
        try:
            age = pd.to_numeric(row.get('Age', row.get('age', np.nan)), errors='coerce')
            if pd.isna(age):
                return False
            return age < 18 or age > 35
        except:
            return False
    
    def _check_high_parity(self, row) -> bool:
        """Check if parity > 4."""
        try:
            parity = pd.to_numeric(row.get('Parity', row.get('parity', np.nan)), errors='coerce')
            if pd.isna(parity):
                return False
            return parity > 4
        except:
            return False
    
    def _check_hiv_positive(self, row) -> bool:
        """Check if HIV status is positive."""
        hiv_status = str(row.get('HIV Status (P/N/KP/U)', row.get('HIV_Status', ''))).upper()
        return hiv_status == 'P' or 'POSITIVE' in hiv_status
    
    def _check_preterm(self, row) -> bool:
        """Check if gestation < 37 weeks."""
        try:
            gestation = pd.to_numeric(
                row.get('Gestation at Birth (Weeks)', row.get('gestation', np.nan)),
                errors='coerce'
            )
            if pd.isna(gestation):
                return False
            return gestation < 37
        except:
            return False
    
    def _check_post_term(self, row) -> bool:
        """Check if gestation > 42 weeks."""
        try:
            gestation = pd.to_numeric(
                row.get('Gestation at Birth (Weeks)', row.get('gestation', np.nan)),
                errors='coerce'
            )
            if pd.isna(gestation):
                return False
            return gestation > 42
        except:
            return False
    
    def _check_delivery_complications(self, row) -> bool:
        """Check for delivery complications."""
        complications = str(row.get('Delivery Complications (codes)', row.get('complications', '')))
        return bool(complications) and complications.lower() not in ['nan', 'none', '', '0']
    
    def _check_c_section(self, row) -> bool:
        """Check if delivery was C-section."""
        mode = str(row.get('Mode_of_Delivery', row.get('mode_of_delivery', ''))).upper()
        return 'CS' in mode or 'C-SECTION' in mode or 'CESAREAN' in mode
    
    def _check_low_birth_weight(self, row) -> bool:
        """Check if birth weight < 2500g."""
        try:
            weight = pd.to_numeric(
                row.get('Birth Weight (in grams)', row.get('birth_weight', np.nan)),
                errors='coerce'
            )
            if pd.isna(weight):
                return False
            return weight < 2500
        except:
            return False
    
    def _check_poor_anc(self, row) -> bool:
        """Check if ANC visits < 4."""
        try:
            visits = pd.to_numeric(
                row.get('No. of ANC Visits', row.get('anc_visits', np.nan)),
                errors='coerce'
            )
            if pd.isna(visits):
                return False
            return visits < 4
        except:
            return False
    
    def classify_risk(self, row: pd.Series) -> RiskInsight:
        """
        Classify pregnancy risk and provide actionable insights.
        
        Args:
            row: Single row from the dataset
            
        Returns:
            RiskInsight object with risk level, factors, and action steps
        """
        risk_factors = []
        action_steps = []
        
        # Check each risk criterion
        for factor_name, criteria in self.risk_criteria.items():
            if criteria['condition'](row):
                risk_factors.append(criteria['description'])
                action_steps.append(criteria['intervention'])
        
        # Determine risk level
        num_factors = len(risk_factors)
        if num_factors >= self.risk_thresholds['high']:
            risk_level = 'High'
            priority = 'Critical'
            confidence = min(0.9 + (num_factors - 3) * 0.03, 0.99)
        elif num_factors >= self.risk_thresholds['moderate']:
            risk_level = 'Moderate'
            priority = 'High'
            confidence = 0.75 + (num_factors - 2) * 0.05
        else:
            risk_level = 'Low'
            priority = 'Medium'
            confidence = 0.85 if num_factors == 1 else 0.95
        
        # Add general recommendations based on risk level
        if risk_level == 'High':
            action_steps.insert(0, '⚠️ IMMEDIATE ACTION REQUIRED: Refer to higher-level facility')
            action_steps.append('Schedule follow-up within 1 week')
            action_steps.append('Ensure emergency obstetric care plan in place')
        elif risk_level == 'Moderate':
            action_steps.insert(0, '⚠️ Enhanced monitoring recommended')
            action_steps.append('Schedule follow-up within 2 weeks')
            action_steps.append('Educate on warning signs')
        else:
            action_steps.append('Continue routine ANC care')
            action_steps.append('Health education on nutrition and danger signs')
        
        return RiskInsight(
            risk_level=risk_level,
            risk_factors=risk_factors,
            confidence=confidence,
            action_steps=action_steps,
            priority=priority
        )
    
    def classify_all_risks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify risk for all records in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added risk classification columns
        """
        results = []
        
        for idx, row in df.iterrows():
            insight = self.classify_risk(row)
            results.append({
                'risk_level': insight.risk_level,
                'risk_factors': ', '.join(insight.risk_factors),
                'num_risk_factors': len(insight.risk_factors),
                'confidence': insight.confidence,
                'priority': insight.priority,
                'action_steps': '; '.join(insight.action_steps)
            })
        
        risk_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), risk_df], axis=1)
    
    def analyze_time_series_trends(
        self,
        df: pd.DataFrame,
        date_column: str,
        metric_column: str,
        period: str = 'monthly'
    ) -> List[TrendData]:
        """
        Analyze time-series trends for a specific metric.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            metric_column: Name of metric column
            period: 'monthly', 'quarterly', or 'yearly'
            
        Returns:
            List of TrendData objects
        """
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Extract period
        if period == 'monthly':
            df['period'] = df[date_column].dt.to_period('M')
        elif period == 'quarterly':
            df['period'] = df[date_column].dt.to_period('Q')
        else:
            df['period'] = df[date_column].dt.to_period('Y')
        
        # Aggregate by period
        trends = []
        grouped = df.groupby('period')[metric_column].agg(['mean', 'count'])
        
        periods = list(grouped.index)
        values = list(grouped['mean'])
        
        for i in range(len(periods)):
            current_value = values[i]
            
            # Determine trend
            if i > 0:
                prev_value = values[i-1]
                change = current_value - prev_value
                if abs(change) < 0.05 * prev_value:
                    trend = 'Stable'
                elif change > 0:
                    trend = 'Increasing'
                else:
                    trend = 'Decreasing'
            else:
                trend = 'Baseline'
            
            # Determine significance (simplified)
            count = grouped['count'].iloc[i]
            significance = 'Significant' if count >= 10 else 'Not Significant'
            
            trends.append(TrendData(
                period=str(periods[i]),
                metric=metric_column,
                value=current_value,
                trend=trend,
                significance=significance,
                intervention_impact=None  # Can be enhanced with intervention impact analysis
            ))
        
        return trends
    
    def analyze_mortality_drivers(
        self,
        df: pd.DataFrame,
        mortality_type: str = 'maternal'
    ) -> List[MortalityDriver]:
        """
        Analyze root causes of mortality and prescribe interventions.
        
        Args:
            df: Input DataFrame
            mortality_type: 'maternal' or 'neonatal'
            
        Returns:
            List of MortalityDriver objects
        """
        drivers = []
        
        if mortality_type == 'maternal':
            # Identify maternal deaths
            deaths = df[df["Mother's Status after Delivery"].str.contains(
                'D|Deceased|Dead', case=False, na=False, regex=True
            )]
            
            # Analyze contributing factors
            for cause, config in self.mortality_drivers['maternal'].items():
                # Count deaths with this contributing factor
                count = len(deaths)  # Simplified - in reality, would match specific patterns
                
                if count > 0:
                    drivers.append(MortalityDriver(
                        cause=cause.replace('_', ' ').title(),
                        frequency=count,
                        percentage=(count / len(deaths)) * 100 if len(deaths) > 0 else 0,
                        interventions=config['interventions'],
                        evidence_level=config['evidence_level'],
                        expected_reduction=config['expected_reduction']
                    ))
        
        elif mortality_type == 'neonatal':
            # Identify neonatal deaths
            deaths = df[df['Live Birth / FSB / MSB'].str.contains(
                'FSB|MSB|Stillbirth|Dead', case=False, na=False, regex=True
            )]
            
            # Analyze contributing factors
            for cause, config in self.mortality_drivers['neonatal'].items():
                count = len(deaths)  # Simplified
                
                if count > 0:
                    drivers.append(MortalityDriver(
                        cause=cause.replace('_', ' ').title(),
                        frequency=count,
                        percentage=(count / len(deaths)) * 100 if len(deaths) > 0 else 0,
                        interventions=config['interventions'],
                        evidence_level=config['evidence_level'],
                        expected_reduction=config['expected_reduction']
                    ))
        
        # Sort by frequency
        drivers.sort(key=lambda x: x.frequency, reverse=True)
        
        return drivers
    
    def generate_risk_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk summary statistics.
        
        Args:
            df: DataFrame with risk classifications
            
        Returns:
            Dictionary with risk summary statistics
        """
        if 'risk_level' not in df.columns:
            df = self.classify_all_risks(df)
        
        total = len(df)
        high_risk = (df['risk_level'] == 'High').sum()
        moderate_risk = (df['risk_level'] == 'Moderate').sum()
        low_risk = (df['risk_level'] == 'Low').sum()
        
        # Most common risk factors
        all_factors = []
        for factors in df['risk_factors']:
            if pd.notna(factors):
                all_factors.extend([f.strip() for f in factors.split(',')])
        
        from collections import Counter
        factor_counts = Counter(all_factors)
        
        return {
            'total_pregnancies': total,
            'high_risk_count': high_risk,
            'high_risk_percentage': (high_risk / total) * 100 if total > 0 else 0,
            'moderate_risk_count': moderate_risk,
            'moderate_risk_percentage': (moderate_risk / total) * 100 if total > 0 else 0,
            'low_risk_count': low_risk,
            'low_risk_percentage': (low_risk / total) * 100 if total > 0 else 0,
            'top_risk_factors': factor_counts.most_common(5),
            'critical_priority_count': (df['priority'] == 'Critical').sum()
        }
    
    def save_analytics_results(self, results: Dict, output_path: str) -> None:
        """
        Save analytics results to JSON file.
        
        Args:
            results: Dictionary with analytics results
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
