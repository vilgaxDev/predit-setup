"""
view_figures.py
==============
Script to display all generated figures with descriptions for research paper.
Run this to review all figures before including in your publication.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Define figure information
FIGURES = [
    {
        "file": "analysis_graphs.png",
        "title": "Figure 1: NLP Processing Metrics",
        "category": "NLP & Anonymization",
        "description": "Comprehensive visualization of NLP pipeline quality metrics showing linearisation coverage rate, delimiter consistency, tokenisation statistics, and BIO tagging performance.",
        "metrics": ["Coverage Rate: 100%", "Delimiter Consistency: 100%", "Total Tokens: 20,000+", "Entity Coverage: 100%"],
        "paper_section": "Methods - Data Preprocessing"
    },
    {
        "file": "ner_performance_metrics.png",
        "title": "Figure 2: NER Performance Metrics",
        "category": "NLP & Anonymization",
        "description": "Bar chart comparing precision, recall, F1 score, and accuracy for Named Entity Recognition. Shows performance against 90% target benchmark.",
        "metrics": ["Precision: 91%", "Recall: 90%", "F1 Score: 90%", "Accuracy: 94%"],
        "paper_section": "Results - Model Performance"
    },
    {
        "file": "data_distribution.png",
        "title": "Figure 3: Data Distribution Analysis",
        "category": "Clinical Data Distribution",
        "description": "Distribution plots for key clinical indicators: Age, Birth Weight, and Gestation period. Shows normality and outliers in the dataset.",
        "metrics": ["333 records", "43 fields per record"],
        "paper_section": "Results - Dataset Characteristics"
    },
    {
        "file": "gender_distribution.png",
        "title": "Figure 4: Gender Distribution",
        "category": "Clinical Data Distribution",
        "description": "Pie chart showing gender distribution in the dataset (male/female infants).",
        "metrics": [],
        "paper_section": "Results - Demographics"
    },
    {
        "file": "delivery_mode.png",
        "title": "Figure 5: Delivery Mode Distribution",
        "category": "Clinical Data Distribution",
        "description": "Bar chart showing distribution of delivery modes (Normal, C-Section, Assisted, etc.).",
        "metrics": [],
        "paper_section": "Results - Clinical Practice"
    },
    {
        "file": "hiv_status.png",
        "title": "Figure 6: HIV Status Distribution",
        "category": "Clinical Data Distribution",
        "description": "Pie chart showing HIV status distribution (Positive, Negative, Known Positive, Unknown).",
        "metrics": [],
        "paper_section": "Results - PMTCT Indicators"
    },
    {
        "file": "marital_status.png",
        "title": "Figure 7: Marital Status Distribution",
        "category": "Clinical Data Distribution",
        "description": "Bar chart showing marital status distribution among mothers.",
        "metrics": [],
        "paper_section": "Results - Social Determinants"
    },
    {
        "file": "diagnosis_distribution.png",
        "title": "Figure 8: Diagnosis Distribution",
        "category": "Clinical Data Distribution",
        "description": "Bar chart showing top 10 most common diagnoses.",
        "metrics": [],
        "paper_section": "Results - Clinical Burden"
    },
    {
        "file": "parity_distribution.png",
        "title": "Figure 9: Parity Distribution",
        "category": "Clinical Data Distribution",
        "description": "Histogram showing parity distribution (number of previous births).",
        "metrics": [],
        "paper_section": "Results - Risk Factors"
    },
    {
        "file": "gravidae_distribution.png",
        "title": "Figure 10: Gravidae Distribution",
        "category": "Clinical Data Distribution",
        "description": "Histogram showing gravidae distribution (total number of pregnancies).",
        "metrics": [],
        "paper_section": "Results - Risk Factors"
    },
    {
        "file": "birth_status.png",
        "title": "Figure 11: Birth Status Distribution",
        "category": "Clinical Data Distribution",
        "description": "Pie chart showing live birth, fresh stillbirth (FSB), and macerated stillbirth (MSB) rates.",
        "metrics": [],
        "paper_section": "Results - Neonatal Outcomes"
    },
    {
        "file": "mother_status.png",
        "title": "Figure 12: Mother's Status After Delivery",
        "category": "Clinical Data Distribution",
        "description": "Pie chart showing maternal survival status after delivery.",
        "metrics": [],
        "paper_section": "Results - Maternal Outcomes"
    },
    {
        "file": "apgar_score.png",
        "title": "Figure 13: APGAR Score Distribution",
        "category": "Health Indicators",
        "description": "Histogram showing APGAR score distribution at 5 minutes.",
        "metrics": [],
        "paper_section": "Results - Newborn Health"
    },
    {
        "file": "anc_visits.png",
        "title": "Figure 14: ANC Visits Distribution",
        "category": "Health Indicators",
        "description": "Histogram showing number of ANC visits distribution.",
        "metrics": [],
        "paper_section": "Results - Antenatal Care"
    },
    {
        "file": "high_risk_distribution.png",
        "title": "Figure 15: High-Risk Pregnancy Distribution",
        "category": "Risk Analysis",
        "description": "Pie chart showing high-risk vs low-risk pregnancy classification based on WHO criteria.",
        "metrics": ["WHO Criteria Applied"],
        "paper_section": "Results - Risk Stratification"
    },
    {
        "file": "risk_factors.png",
        "title": "Figure 16: Risk Factors Distribution",
        "category": "Risk Analysis",
        "description": "Bar chart showing frequency of each risk factor (age extremes, high parity, HIV positive, etc.).",
        "metrics": ["9 WHO Risk Factors"],
        "paper_section": "Results - Risk Factor Prevalence"
    },
    {
        "file": "maternal_mortality_trend.png",
        "title": "Figure 17: Maternal Mortality Trend",
        "category": "Temporal Analysis",
        "description": "Line graph showing maternal mortality rate trend over years with trend analysis (increasing/decreasing/stable).",
        "metrics": ["Rate per 1,000 deliveries"],
        "paper_section": "Results - Temporal Analysis"
    },
    {
        "file": "neonatal_mortality_trend.png",
        "title": "Figure 18: Neonatal Mortality Trend",
        "category": "Temporal Analysis",
        "description": "Line graph showing neonatal mortality rate trend over years with trend analysis.",
        "metrics": ["Rate per 1,000 deliveries"],
        "paper_section": "Results - Temporal Analysis"
    },
    {
        "file": "knowledge_graph_structure.png",
        "title": "Figure 19: Knowledge Graph Structure",
        "category": "NLP & Anonymization",
        "description": "Network graph visualization showing patient nodes and their relationships via RDF triples (sample of 200 triples).",
        "metrics": ["14,000+ triples", "43 unique predicates"],
        "paper_section": "Methods - KG Architecture"
    },
    {
        "file": "knowledge_graph_predicates.png",
        "title": "Figure 20: Knowledge Graph Predicates",
        "category": "NLP & Anonymization",
        "description": "Bar chart showing distribution of KG predicates (field types) across all triples.",
        "metrics": ["Top 10 predicates shown"],
        "paper_section": "Results - KG Integration"
    },
    {
        "file": "metrics_comparison.png",
        "title": "Figure 21: Metrics Comparison",
        "category": "Model Performance",
        "description": "Comparison chart showing before/after metrics improvement.",
        "metrics": [],
        "paper_section": "Results - Framework Effectiveness"
    }
]

def display_figure_info():
    """Display information about all figures."""
    print("=" * 80)
    print("FIGURE CATALOG FOR RESEARCH PAPER")
    print("=" * 80)
    print(f"Total Figures: {len(FIGURES)}")
    print(f"Location: {Path(__file__).parent}")
    print()
    
    # Group by category
    categories = {}
    for fig in FIGURES:
        cat = fig["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(fig)
    
    # Display by category
    for category, figures in categories.items():
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 80}")
        
        for fig in figures:
            print(f"\n{fig['title']}")
            print(f"  File: {fig['file']}")
            print(f"  Section: {fig['paper_section']}")
            print(f"  Description: {fig['description']}")
            if fig['metrics']:
                print(f"  Key Metrics: {', '.join(fig['metrics'])}")
    
    print("\n" + "=" * 80)
    print("FIGURE SUMMARY FOR PAPER")
    print("=" * 80)
    print("\nSuggested Figure Allocation:")
    print("-" * 80)
    
    # Main figures (most important)
    main_figures = [1, 2, 15, 17, 18, 19]
    print("\nMAIN FIGURES (6 figures):")
    for i in main_figures:
        fig = FIGURES[i-1]
        print(f"  {fig['title']} - {fig['paper_section']}")
    
    # Supplementary figures
    supplementary = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20, 21]
    print("\nSUPPLEMENTARY FIGURES (15 figures):")
    for i in supplementary:
        fig = FIGURES[i-1]
        print(f"  {fig['title']} - {fig['paper_section']}")

def check_figures_exist():
    """Check which figures exist."""
    print("\n" + "=" * 80)
    print("FIGURE AVAILABILITY CHECK")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    missing = []
    existing = []
    
    for fig in FIGURES:
        filepath = base_dir / fig["file"]
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            existing.append((fig, size_kb))
        else:
            missing.append(fig)
    
    print(f"\nExisting Figures: {len(existing)}/{len(FIGURES)}")
    for fig, size in existing:
        print(f"  ✓ {fig['file']} ({size:.1f} KB)")
    
    if missing:
        print(f"\nMissing Figures: {len(missing)}/{len(FIGURES)}")
        for fig in missing:
            print(f"  ✗ {fig['file']}")
        print("\nTo regenerate missing figures, run:")
        print("  python run_notebook.py")
    
    return len(missing) == 0

def generate_figure_list():
    """Generate a text file with figure list for paper."""
    output_file = Path(__file__).parent / "FIGURE_LIST.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FIGURE LIST FOR RESEARCH PAPER\n")
        f.write("=" * 80 + "\n\n")
        
        for i, fig in enumerate(FIGURES, 1):
            f.write(f"Figure {i}: {fig['title']}\n")
            f.write(f"File: {fig['file']}\n")
            f.write(f"Category: {fig['category']}\n")
            f.write(f"Paper Section: {fig['paper_section']}\n")
            f.write(f"Description: {fig['description']}\n")
            if fig['metrics']:
                f.write(f"Key Metrics: {', '.join(fig['metrics'])}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"\n✓ Figure list saved to: {output_file}")

def main():
    """Main function."""
    display_figure_info()
    all_exist = check_figures_exist()
    generate_figure_list()
    
    if all_exist:
        print("\n✓ All figures are available and ready for use!")
        print("\nNext steps:")
        print("  1. Review figures in dashboard: streamlit run app.py")
        print("  2. Select main figures for main paper")
        print("  3. Use remaining figures as supplementary material")
        print("  4. Regenerate if needed: python run_notebook.py")
    else:
        print("\n⚠ Some figures are missing. Run python run_notebook.py to regenerate.")

if __name__ == "__main__":
    main()
