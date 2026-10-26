"""
pmtct_tracker.py
================
Longitudinal Prevention of Mother-to-Child Transmission (PMTCT) and
Maternal HIV Seroconversion Incidence tracker for Kenyan health records (MOH 206).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_seroconversion_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies maternal HIV seroconversions (documented negative baseline -> confirmed positive retest).
    Distinguishes confirmed cases from suspected/incomplete, and calculates incidence rates.
    """
    eligible_baseline_neg = df[df['hiv_status_at_enrolment'] == 'Negative']
    total_baseline_neg = len(eligible_baseline_neg)

    confirmed_seroconversions = df[
        (df['hiv_status_at_enrolment'] == 'Negative') &
        (df['seroconversion_status'] == 'Confirmed Seroconversion')
    ]
    
    suspected_seroconversions = df[
        (df['hiv_status_at_enrolment'] == 'Negative') &
        (df['seroconversion_status'] == 'Suspected / Discordant')
    ]

    confirmed_count = len(confirmed_seroconversions)
    suspected_count = len(suspected_seroconversions)

    # Incidence per 1,000 eligible women
    incidence_per_1000 = (confirmed_count / total_baseline_neg * 1000) if total_baseline_neg > 0 else 0.0

    return {
        'total_eligible_baseline_negative': total_baseline_neg,
        'confirmed_seroconversions': confirmed_count,
        'suspected_seroconversions': suspected_count,
        'incidence_per_1000_pregnant_women': round(incidence_per_1000, 2),
        'by_county': confirmed_seroconversions.groupby('county').size().to_dict(),
        'by_detection_period': confirmed_seroconversions.groupby('seroconversion_detection_period').size().to_dict() if 'seroconversion_detection_period' in confirmed_seroconversions.columns else {}
    }

def calculate_pmtct_five_year_cascade(pmtct_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates 5-year pediatric HIV-free survival, maternal survival,
    testing completeness, and loss to follow-up (LTFU) across exposed infants.
    """
    total_exposed = len(pmtct_df)
    if total_exposed == 0:
        return {}

    confirmed_negative_survived = len(pmtct_df[pmtct_df['five_year_child_hiv_status'] == 'Confirmed HIV-Negative'])
    pediatric_transmissions = len(pmtct_df[pmtct_df['five_year_child_hiv_status'] == 'Confirmed HIV-Positive'])
    lost_to_followup = len(pmtct_df[pmtct_df['five_year_child_hiv_status'] == 'Lost to Follow-up'])
    child_deaths = len(pmtct_df[pmtct_df['five_year_child_hiv_status'] == 'Child Deceased'])
    cohort_in_progress = len(pmtct_df[pmtct_df['five_year_child_hiv_status'] == 'Cohort In-Progress'])

    # Viral suppression distribution among mothers
    viral_counts = pmtct_df['viral_suppression_category'].value_counts().to_dict()

    return {
        'total_hiv_exposed_infants': total_exposed,
        'confirmed_hiv_negative_survived_5yr': confirmed_negative_survived,
        'confirmed_pediatric_transmissions': pediatric_transmissions,
        'lost_to_follow_up_5yr': lost_to_followup,
        'child_mortality_5yr': child_deaths,
        'cohort_active_in_progress': cohort_in_progress,
        'hiv_free_survival_rate_pct': round((confirmed_negative_survived / (total_exposed - cohort_in_progress) * 100), 1) if (total_exposed - cohort_in_progress) > 0 else 0.0,
        'maternal_viral_suppression': viral_counts
    }
