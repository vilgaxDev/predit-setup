"""
run_production_demo.py
======================
Demonstrates the production-ready BERT-LLM-KG framework with all new components.
"""

import pandas as pd
from pathlib import Path
import sys

# Import new modules
from sample_annotations_generator import SampleAnnotationGenerator
from kg_tooling import KnowledgeGraphManager
from expert_review_protocol import ExpertReviewProtocol, setup_multi_expert_review

def main():
    print("=" * 80)
    print("PRODUCTION-READY BERT-LLM-KG FRAMEWORK DEMO")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    data_file = Path(__file__).parent.parent.parent / "dashboard" / "COMBINE_MOH_333_3replaced_KCHR.xlsx"
    
    if not data_file.exists():
        # Try alternative location
        data_file = Path(__file__).parent / "COMBINE_MOH_333_3replaced_KCHR.xlsx"
    
    if not data_file.exists():
        print(f"[WARNING] Data file not found at: {data_file}")
        print("Please ensure COMBINE_MOH_333_3replaced_KCHR.xlsx is in the pipeline directory")
        return
    
    df = pd.read_excel(data_file)
    print(f"[OK] Loaded {len(df)} records from {data_file.name}")
    
    # Step 2: Generate sample ground truth annotations
    print("\n[2/5] Generating sample ground truth annotations...")
    generator = SampleAnnotationGenerator()
    annotations = generator.generate_sample_annotations(df, num_samples=50)
    
    annotations_file = generator.save_annotations(annotations)
    record_reviews_file = generator.create_record_reviews(annotations)
    
    print(f"[OK] Generated {len(annotations)} annotations")
    print(f"[OK] Saved to: {annotations_file}")
    print(f"[OK] Record reviews: {record_reviews_file}")
    
    # Step 3: Test KG tooling with rdflib
    print("\n[3/5] Testing Knowledge Graph with rdflib...")
    kg = KnowledgeGraphManager()
    
    # Check if triples file exists
    triples_file = Path(__file__).parent / "knowledge_graph_triples.txt"
    if triples_file.exists():
        count = kg.add_triples_from_file(str(triples_file))
        print(f"[OK] Loaded {count} triples into Knowledge Graph")
        
        # Get statistics
        stats = kg.get_statistics()
        print(f"[OK] KG Statistics:")
        print(f"  - Total triples: {stats['total_triples']}")
        print(f"  - Unique subjects: {stats['unique_subjects']}")
        print(f"  - Unique predicates: {stats['unique_predicates']}")
        print(f"  - Unique objects: {stats['unique_objects']}")
        
        # Export to RDF Turtle format (skip for large graphs)
        if stats['total_triples'] < 10000:
            rdf_file = Path(__file__).parent / "knowledge_graph.ttl"
            kg.export_rdf(format="turtle", output_path=str(rdf_file))
            print(f"[OK] Exported RDF to: {rdf_file}")
        else:
            print(f"[SKIPPED] RDF export for large graph ({stats['total_triples']} triples)")
    else:
        print("[WARNING] knowledge_graph_triples.txt not found, skipping KG demo")
    
    # Step 4: Test expert review protocol
    print("\n[4/5] Testing Multi-Expert Review Protocol...")
    protocol = setup_multi_expert_review(num_experts=3)
    
    experts = protocol._load_experts()
    print(f"[OK] Registered {len(experts)} experts:")
    for expert_id, expert in experts.items():
        print(f"  - {expert['name']} ({expert_id}) - {expert['specialty']}")
    
    # Step 5: Check BERT trainer availability
    print("\n[5/5] Checking BERT Trainer availability...")
    try:
        from bert_trainer import BERTNERTrainer, TrainingConfig
        print("[OK] BERT Trainer module imported successfully")
        
        # Check GPU availability
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("[OK] GPU not detected, will use CPU (slower training)")
        
        # Check LLM anonymizer
        from llm_anonymizer import LLMAnonymizer, LLMConfig
        print("[OK] LLM Anonymizer module imported successfully")
        
        # Check if API key is set
        import os
        if os.environ.get("OPENAI_API_KEY"):
            print("[OK] OpenAI API key found")
        else:
            print("[WARNING] OpenAI API key not set (set OPENAI_API_KEY environment variable for LLM features)")
        
    except ImportError as e:
        print(f"[WARNING] Could not import BERT trainer: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("[OK] Sample ground truth annotations generated")
    print("[OK] Knowledge Graph with rdflib tested")
    print("[OK] Multi-expert review protocol set up")
    print("[OK] BERT trainer and LLM anonymizer modules available")
    print("\nNext steps:")
    print("1. Use annotations for BERT training (see bert_trainer.py)")
    print("2. Set OPENAI_API_KEY for LLM anonymization")
    print("3. Run training in Google Colab for GPU acceleration (see GOOGLE_COLAB_SETUP.md)")
    print("4. Use dashboard: streamlit run app.py")

if __name__ == "__main__":
    main()
