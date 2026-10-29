"""
run_multi_dataset_evaluation.py
=================================
Script to run multi-dataset evaluation for ANC, PNC, and MOH 204A registers.
Demonstrates expert template fact extraction and collective evaluation.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Import evaluation modules
from expert_template_manager import ExpertTemplateManager
from multi_dataset_evaluator import MultiDatasetEvaluator
from dataset_specific_evaluator import DatasetSpecificEvaluator

# Set up paths
DATA_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent.parent.parent

def find_data_file(filename, environment_variable):
    """Find data file in multiple locations."""
    configured_path = os.environ.get(environment_variable)
    candidates = [configured_path] if configured_path else []
    candidates.extend([
        os.path.join(DATA_DIR, filename),
        os.path.join(PROJECT_DIR, filename)
    ])
    resolved = next((path for path in candidates if os.path.isfile(path)), candidates[-1])
    return resolved

def main():
    """Main evaluation workflow."""
    print("=" * 80)
    print("MULTI-DATASET EVALUATION: ANC, PNC, and MOH 204A")
    print("=" * 80)
    print()
    
    # Initialize evaluators
    template_manager = ExpertTemplateManager()
    multi_evaluator = MultiDatasetEvaluator()
    dataset_evaluator = DatasetSpecificEvaluator()
    
    # Step 1: Load datasets
    print("Step 1: Loading Datasets")
    print("-" * 80)
    
    files = {
        'anc': find_data_file("MOH_405_ANC final JAN 2026.xlsx", "MCH_ANC_FILE"),
        'pnc': find_data_file("MOH_406_PNC.xlsx", "MCH_PNC_FILE"),
        'under5': find_data_file("MOH_204A_UNDER5 revised version on jan 2026.xlsx", "MCH_UNDER5_FILE"),
        'maternity': find_data_file("COMBINE_MOH_333_3replaced_KCHR.xlsx", "MCH_MATERNITY_FILE")
    }
    
    datasets = {}
    for name, filepath in files.items():
        if os.path.exists(filepath):
            print(f"Loading {name.upper()} from: {filepath}")
            try:
                if filepath.lower().endswith(".csv"):
                    datasets[name] = pd.read_csv(filepath)
                else:
                    datasets[name] = pd.read_excel(filepath, engine="openpyxl")
                print(f"  ✓ Loaded {len(datasets[name])} records")
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")
                datasets[name] = None
        else:
            print(f"  ✗ File not found for {name}: {filepath}")
            datasets[name] = None
    
    print()
    
    # Step 2: Display available templates
    print("Step 2: Available Expert Templates")
    print("-" * 80)
    templates = template_manager.list_templates()
    for template in templates:
        print(f"  {template['template_id']}: {template['template_name']}")
        print(f"    Register Type: {template['register_type']}")
        print(f"    Fields: {template['field_count']}")
        print(f"    Description: {template['description']}")
        print()
    
    # Step 3: Dataset-specific evaluation
    print("Step 3: Dataset-Specific Evaluation")
    print("-" * 80)
    
    if datasets.get('anc') is not None:
        print("\nANC Register Evaluation:")
        anc_results = dataset_evaluator.evaluate_anc_register(
            datasets['anc'],
            "MOH 405 ANC Register"
        )
        print(f"  Total Records: {anc_results.total_records}")
        print(f"  Valid Records: {anc_results.valid_records}")
        print(f"  Validation Rate: {anc_results.validation_rate:.2%}")
        print(f"  Average ANC Visits: {anc_results.average_anc_visits:.2f}")
        print(f"  HIV Positive Rate: {anc_results.hiv_positive_rate:.2%}")
        print(f"  Hypertension Detection Rate: {anc_results.hypertension_detection_rate:.2%}")
        print(f"  Anemia Detection Rate: {anc_results.anemia_detection_rate:.2%}")
        print(f"  BP Measurement Completeness: {anc_results.bp_measurement_completeness:.2%}")
        print(f"  Hemoglobin Completeness: {anc_results.hemoglobin_completeness:.2%}")
        print(f"  First Trimester Booking Rate: {anc_results.first_trimester_booking_rate:.2%}")
        print(f"  Tetanus Coverage: {anc_results.tetanus_coverage:.2%}")
        print(f"  IPT Coverage: {anc_results.ipt_coverage:.2%}")
    
    if datasets.get('pnc') is not None:
        print("\nPNC Register Evaluation:")
        pnc_results = dataset_evaluator.evaluate_pnc_register(
            datasets['pnc'],
            "MOH 406 PNC Register"
        )
        print(f"  Total Records: {pnc_results.total_records}")
        print(f"  Valid Records: {pnc_results.valid_records}")
        print(f"  Validation Rate: {pnc_results.validation_rate:.2%}")
        print(f"  Maternal Survival Rate: {pnc_results.maternal_survival_rate:.2%}")
        print(f"  Neonatal Survival Rate: {pnc_results.neonatal_survival_rate:.2%}")
        print(f"  Early PNC Visit Rate: {pnc_results.early_pnc_visit_rate:.2%}")
        print(f"  Breastfeeding Initiation Rate: {pnc_results.breastfeeding_initiation_rate:.2%}")
        print(f"  Postpartum Hemorrhage Rate: {pnc_results.postpartum_hemorrhage_rate:.2%}")
        print(f"  PNC Visit Completeness: {pnc_results.pnc_visit_completeness:.2%}")
        print(f"  Follow-up Adherence: {pnc_results.follow_up_adherence:.2%}")
    
    if datasets.get('under5') is not None:
        print("\nUnder-5 Register Evaluation:")
        under5_results = dataset_evaluator.evaluate_under5_register(
            datasets['under5'],
            "MOH 204A Under-5 Register"
        )
        print(f"  Total Records: {under5_results.total_records}")
        print(f"  Valid Records: {under5_results.valid_records}")
        print(f"  Validation Rate: {under5_results.validation_rate:.2%}")
        print(f"  Severe Acute Malnutrition Rate: {under5_results.malnutrition_severe_rate:.2%}")
        print(f"  Moderate Acute Malnutrition Rate: {under5_results.malnutrition_moderate_rate:.2%}")
        print(f"  Underweight Rate: {under5_results.underweight_rate:.2%}")
        print(f"  Stunting Rate: {under5_results.stunting_rate:.2%}")
        print(f"  Vaccination Complete Rate: {under5_results.vaccination_complete_rate:.2%}")
        print(f"  Growth Monitoring Completeness: {under5_results.growth_monitoring_completeness:.2%}")
        print(f"  MUAC Measurement Completeness: {under5_results.muac_measurement_completeness:.2%}")
    
    print()
    
    # Step 4: Fact extraction using expert templates
    print("Step 4: Fact Extraction Using Expert Templates")
    print("-" * 80)
    
    datasets_for_extraction = {
        'anc': (datasets.get('anc'), 'ANC'),
        'pnc': (datasets.get('pnc'), 'PNC'),
        'under5': (datasets.get('under5'), 'UNDER5')
    }
    
    for dataset_name, (df, dataset_type) in datasets_for_extraction.items():
        if df is not None:
            print(f"\nExtracting facts from {dataset_name.upper()} register...")
            try:
                template = template_manager.get_template_by_register(dataset_type)
                if template:
                    facts = template_manager.extract_facts_from_dataset(df, template['template_id'])
                    print(f"  ✓ Extracted {len(facts)} facts")
                    
                    # Convert to KG triples
                    kg_triples = template_manager.convert_facts_to_kg_triples(facts, template)
                    print(f"  ✓ Generated {len(kg_triples)} RDF triples")
                    
                    # Save facts
                    template_manager.save_extracted_facts(facts)
                    print(f"  ✓ Saved facts to expert_templates/extracted_facts.jsonl")
                    
                    # Display sample facts
                    print(f"\n  Sample facts (first 5):")
                    for fact in facts[:5]:
                        print(f"    - {fact['field_name']}: {fact['value']} ({fact['fact_type']})")
                else:
                    print(f"  ✗ No template found for {dataset_type}")
            except Exception as e:
                print(f"  ✗ Error extracting facts: {e}")
    
    print()
    
    # Step 5: Collective evaluation
    print("Step 5: Collective Multi-Dataset Evaluation")
    print("-" * 80)
    
    datasets_for_eval = {}
    for dataset_name, (df, dataset_type) in datasets_for_extraction.items():
        if df is not None:
            datasets_for_eval[f"{dataset_name}_register"] = (df, dataset_type)
    
    if datasets_for_eval:
        print(f"Evaluating {len(datasets_for_eval)} datasets collectively...")
        try:
            collective_metrics = multi_evaluator.evaluate_collective(datasets_for_eval)
            
            # Generate report
            report = multi_evaluator.generate_evaluation_report(collective_metrics)
            print(report)
            
            # Save results
            output_path = multi_evaluator.save_evaluation_results(collective_metrics)
            print(f"\n✓ Evaluation results saved to: {output_path}")
            
            # Comparison
            if len(collective_metrics.dataset_metrics) >= 2:
                print("\nDataset Comparison:")
                print("-" * 80)
                comparison = multi_evaluator.compare_datasets(collective_metrics)
                
                if 'best_extraction_rate' in comparison:
                    best = comparison['best_extraction_rate']
                    print(f"Best Extraction Rate: {best.dataset_name} ({best.extraction_rate:.2%})")
                
                if 'best_quality_score' in comparison:
                    best = comparison['best_quality_score']
                    print(f"Best Quality Score: {best.dataset_name} ({best.quality_score:.2%})")
                
                if 'most_facts_extracted' in comparison:
                    best = comparison['most_facts_extracted']
                    print(f"Most Facts Extracted: {best.dataset_name} ({best.facts_extracted:,})")
                
                if 'most_kg_triples' in comparison:
                    best = comparison['most_kg_triples']
                    print(f"Most KG Triples: {best.dataset_name} ({best.kg_triples_generated:,})")
        
        except Exception as e:
            print(f"✗ Error in collective evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No datasets available for collective evaluation")
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("Generated Files:")
    print("  - expert_templates/templates.json (Template definitions)")
    print("  - expert_templates/extracted_facts.jsonl (Extracted facts)")
    print("  - evaluation_results/multi_dataset_evaluation.json (Evaluation results)")
    print()
    print("Next Steps:")
    print("  1. Review evaluation results in the JSON file")
    print("  2. Update dashboard to display multi-dataset metrics")
    print("  3. Add manual annotations for NER evaluation (if desired)")
    print("  4. Customize expert templates for your specific needs")

if __name__ == "__main__":
    main()
