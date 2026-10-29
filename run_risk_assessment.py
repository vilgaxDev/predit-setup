"""
run_risk_assessment.py
=======================
Run re-identification risk assessment on anonymized data.
"""

import pandas as pd
from pathlib import Path
import json
from re_identification_risk import ReIdentificationRiskAssessment, assess_re_identification_risk

def main():
    print("=" * 80)
    print("RE-IDENTIFICATION RISK ASSESSMENT")
    print("=" * 80)
    
    # Load original data
    print("\n[1/4] Loading original data...")
    original_file = Path(__file__).parent / "COMBINE_MOH_333_3replaced_KCHR.xlsx"
    
    if not original_file.exists():
        print(f"[ERROR] Original data file not found: {original_file}")
        return
    
    original_df = pd.read_excel(original_file)
    print(f"[OK] Loaded {len(original_df)} original records")
    
    # Load anonymized data
    print("\n[2/4] Loading anonymized data...")
    anonymized_file = Path(__file__).parent / "anonymized_data_kg_export.xlsx"
    
    if not anonymized_file.exists():
        print(f"[WARNING] Anonymized data file not found: {anonymized_file}")
        print("Using original data as baseline (no anonymization applied)")
        anonymized_df = original_df.copy()
    else:
        anonymized_df = pd.read_excel(anonymized_file)
        print(f"[OK] Loaded {len(anonymized_df)} anonymized records")
    
    # Run risk assessment
    print("\n[3/4] Running risk assessment...")
    assessor = ReIdentificationRiskAssessment(anonymized_df, original_df)
    
    print(f"\nQuasi-identifiers identified: {assessor.quasi_identifiers}")
    print(f"Sensitive attributes identified: {assessor.sensitive_attributes}")
    
    # k-anonymity
    print("\n--- k-Anonymity Assessment ---")
    k_results = assessor.calculate_k_anonymity(k=5)
    if "error" not in k_results:
        print(f"k-value: {k_results['k_value']}")
        print(f"Total equivalence classes: {k_results['total_equivalence_classes']}")
        print(f"Violating classes: {k_results['violating_classes']}")
        print(f"Violation rate: {k_results['violation_rate']:.2f}%")
        print(f"Minimum class size: {k_results['min_class_size']}")
        print(f"Compliant: {k_results['k_anonymity_compliant']}")
    else:
        print(f"Error: {k_results['error']}")
    
    # l-diversity
    print("\n--- l-Diversity Assessment ---")
    l_results = assessor.calculate_l_diversity(l=3)
    if "error" not in l_results:
        for attr, results in l_results.items():
            print(f"\nAttribute: {attr}")
            print(f"  l-value: {results['l_value']}")
            print(f"  Violating classes: {results['violating_classes']}")
            print(f"  Violation rate: {results['violation_rate']:.2f}%")
            print(f"  Minimum diversity: {results['min_diversity']}")
            print(f"  Compliant: {results['l_diversity_compliant']}")
    else:
        print(f"Error: {l_results['error']}")
    
    # t-closeness
    print("\n--- t-Closeness Assessment ---")
    t_results = assessor.calculate_t_closeness(t=0.25)
    if "error" not in t_results:
        for attr, results in t_results.items():
            print(f"\nAttribute: {attr}")
            print(f"  t-value: {results['t_value']}")
            print(f"  Max distance: {results['max_distance']:.4f}")
            print(f"  Compliant: {results['t_closeness_compliant']}")
    else:
        print(f"Error: {t_results['error']}")
    
    # Uniqueness
    print("\n--- Uniqueness Assessment ---")
    uniqueness = assessor.check_uniqueness()
    if "error" not in uniqueness:
        print(f"Total records: {uniqueness['total_records']}")
        print(f"Unique combinations: {uniqueness['unique_combinations']}")
        print(f"Unique records: {uniqueness['unique_records_count']}")
        print(f"Unique rate: {uniqueness['unique_rate']:.2f}%")
    else:
        print(f"Error: {uniqueness['error']}")
    
    # Linkage attack simulation
    print("\n--- Linkage Attack Simulation ---")
    linkage = assessor.simulate_linkage_attack()
    if "error" not in linkage:
        print(f"Total attempts: {linkage['total_attempts']}")
        print(f"Successful linkages: {linkage['successful_linkages']}")
        print(f"Linkage rate: {linkage['linkage_rate']:.2f}%")
        print(f"Linkage risk: {linkage['linkage_risk']}")
    else:
        print(f"Error: {linkage['error']}")
    
    # Generate comprehensive report
    print("\n[4/4] Generating comprehensive report...")
    output_dir = Path(__file__).parent / "risk_assessment_results"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "risk_assessment_report.json"
    report = assessor.generate_risk_report(str(report_file))
    
    # Generate visualizations
    viz_dir = output_dir / "visualizations"
    assessor.visualize_risk_assessment(str(viz_dir))
    
    print("\n" + "=" * 80)
    print("RISK ASSESSMENT COMPLETE")
    print("=" * 80)
    print(f"\nOverall Risk Level: {report['risk_level']}")
    print(f"Risk Score: {report['overall_risk_score']}/100")
    
    if report['risk_factors']:
        print("\nRisk Factors:")
        for factor in report['risk_factors']:
            print(f"  - {factor}")
    else:
        print("\nNo significant risk factors identified.")
    
    print(f"\nReport saved to: {report_file}")
    print(f"Visualizations saved to: {viz_dir}")
    
    print("\nInterpretation:")
    if report['risk_level'] == 'LOW':
        print("  - Data has low re-identification risk")
        print("  - Anonymization is effective")
    elif report['risk_level'] == 'MEDIUM':
        print("  - Data has moderate re-identification risk")
        print("  - Consider additional anonymization")
    else:
        print("  - Data has high re-identification risk")
        print("  - Strongly recommend additional anonymization")

if __name__ == "__main__":
    main()
