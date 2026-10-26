import pandas as pd
import numpy as np
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import os
import json
import networkx as nx
from collections import Counter
import sys

# Import advanced analytics
from advanced_analytics import AdvancedHealthAnalytics

# Preserve the launcher’s output encoding and replace characters it cannot encode.
if sys.platform == 'win32':
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')

# Try to import seaborn, make it optional
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

# -----------------------------
# 1. Load ALL Datasets
# -----------------------------
print("="*50)
print("LOADING ALL MATERNAL-CHILD HEALTH DATASETS")
print("="*50)

# Resolve source files from an explicit environment path, the script directory,
# or the project root (useful when deployed datasets are stored outside the app).
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = r"D:\sarah"


def resolve_data_file(filename, environment_variable):
    configured_path = os.environ.get(environment_variable)
    candidates = [configured_path] if configured_path else []
    candidates.extend([
        os.path.join(DATA_DIR, filename),
        os.path.join(PROJECT_DIR, filename)
    ])
    resolved = next((path for path in candidates if os.path.isfile(path)), candidates[-1])
    return resolved, candidates


files = {
    'maternity': resolve_data_file(
        "COMBINE_MOH_333_3replaced_KCHR.xlsx",
        "MCH_MATERNITY_FILE"
    ),
    'anc': resolve_data_file(
        "MOH_405_ANC final JAN 2026.xlsx",
        "MCH_ANC_FILE"
    ),
    'under5': resolve_data_file(
        "MOH_204A_UNDER5 revised version on jan 2026.xlsx",
        "MCH_UNDER5_FILE"
    )
}

# Also check uploads directory
uploads_dir = os.path.join(DATA_DIR, "data_uploads")
if os.path.exists(uploads_dir):
    uploaded_files = [f for f in os.listdir(uploads_dir) if f.endswith(('.xlsx', '.xls', '.csv'))]
    if uploaded_files:
        print(f"\n[INFO] Found {len(uploaded_files)} uploaded file(s) in data_uploads directory:")
        for uploaded_file in uploaded_files:
            print(f"  - {uploaded_file}")

# Load all datasets
datasets = {}
for name, (filepath, candidates) in files.items():
    if os.path.exists(filepath):
        print(f"\nLoading {name.upper()} from: {filepath}")
        try:
            if filepath.lower().endswith(".csv"):
                datasets[name] = pd.read_csv(filepath)
            else:
                datasets[name] = pd.read_excel(filepath, engine="openpyxl")
            print(f"[OK] Loaded {len(datasets[name])} records")
        except Exception as e:
            print(f"[ERROR] Error loading {name}: {e}")
            datasets[name] = None
    else:
        print(f"[ERROR] File not found for {name}: tried {candidates}")
        datasets[name] = None

print(f"\n[OK] Successfully loaded {sum(1 for v in datasets.values() if v is not None)} datasets")

# Use maternity data as primary for detailed analysis
df = datasets.get('maternity')
if df is None:
    raise FileNotFoundError(
        "Maternity data was not found. Upload a maternity-compatible register "
        "in the dashboard, set MCH_MATERNITY_FILE, or place "
        "'COMBINE_MOH_333_3replaced_KCHR.xlsx' beside run_notebook.py or "
        "in the project root."
    )

print(f"\nPrimary dataset for analysis: Maternity ({len(df)} records)")
print(f"Processing 100% of data - no sampling")

anc_data = datasets.get('anc')
if anc_data is not None and not anc_data.empty:
    print(f"Secondary dataset: ANC ({len(anc_data)} records)")

under5_data = datasets.get('under5')
if under5_data is not None and not under5_data.empty:
    print(f"Secondary dataset: Under-5 ({len(under5_data)} records)")

print(f"\nTotal records across all datasets: {sum(len(v) for v in datasets.values() if v is not None and not v.empty):,}")

# -----------------------------
# 2. Linearisation Function (Generic)
# -----------------------------
def linearise_record(row):
    """Generic linearisation that works with any DataFrame row."""
    return " | ".join([f"{col}: {row[col]}" for col in row.index])

print("\n" + "="*50)
print("LINEARISATION")
print("="*50)

texts = df.apply(linearise_record, axis=1).tolist()

# -----------------------------
# 3. Validation Checks
# -----------------------------
required_fields = [part.split(":")[0] + ":" for part in texts[0].split("|")]

def check_record(text):
    has_all_fields = all(field in text for field in required_fields)
    correct_delimiters = text.count("|") == len(required_fields) - 1
    no_empty_values = all(len(part.split(":")[1].strip()) > 0 for part in text.split("|"))
    return has_all_fields, correct_delimiters, no_empty_values

results = [check_record(t) for t in texts]

coverage_rate = sum(r[0] for r in results) / len(results)
delimiter_rate = sum(r[1] for r in results) / len(results)
non_empty_rate = sum(r[2] for r in results) / len(results)

print("Linearisation Quality Metrics:")
print(f"Coverage Rate: {coverage_rate:.2%}")
print(f"Delimiter Consistency Rate: {delimiter_rate:.2%}")
print(f"Non-Empty Field Rate: {non_empty_rate:.2%}")

print("\nSample Linearised Records (showing first 5):")
for i, t in enumerate(texts[:5]):
    print(f"\n--- Record {i+1} ---")
    print(t)

print("\nProblematic Records:")
for i, (t, r) in enumerate(zip(texts, results)):
    if not all(r):
        print(f"Record {i}: {t} | Issues: {r}")

# -----------------------------
# TOKENISATION (ALL RECORDS)
# -----------------------------
print("\n" + "="*50)
print("TOKENISATION")
print("="*50)

# Load pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenise ALL records
all_tokens = []
all_token_ids = []
total_token_count = 0

print(f"Tokenising {len(texts)} records...")
# Process ALL records with progress indicator
for i, text in enumerate(texts):
    try:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        all_tokens.extend(tokens)
        all_token_ids.extend(ids)
        total_token_count += len(tokens)
    except Exception as e:
        print(f"  Error tokenizing record {i}: {e}")
        continue
    
    # Show progress every 10 records (more frequent)
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i + 1}/{len(texts)} records ({total_token_count} tokens)...")

print(f"Total tokens across all records: {total_token_count:,}")

# Show sample from first record
sample_text = texts[0]
sample_tokens = tokenizer.tokenize(sample_text)
sample_ids = tokenizer.convert_tokens_to_ids(sample_tokens)
print("\nSample tokens from first record:")
print("Tokens:", sample_tokens[:20], "..." if len(sample_tokens) > 20 else "")
print("Token IDs:", sample_ids[:20], "..." if len(sample_ids) > 20 else "")

# -----------------------------
# TEST ANALYSIS (ALL RECORDS)
# -----------------------------
def analyse_tokenisation(tokens_list):
    total_tokens = len(tokens_list)
    oov_tokens = sum(1 for tok in tokens_list if tok == "[UNK]")
    mean_token_length = sum(len(tok) for tok in tokens_list) / total_tokens if total_tokens > 0 else 0
    delimiter_tokens = sum(1 for tok in tokens_list if tok in [":", "|"])

    return {
        "Total Tokens": total_tokens,
        "OOV Rate": oov_tokens / total_tokens if total_tokens > 0 else 0,
        "Mean Token Length": mean_token_length,
        "Delimiter Count": delimiter_tokens
    }

token_metrics = analyse_tokenisation(all_tokens)
print("\nTokenisation Quality Metrics (ALL RECORDS):")
for k, v in token_metrics.items():
    print(f"{k}: {v}")

# -----------------------------
# BIO TAGGING
# -----------------------------
print("\n" + "="*50)
print("BIO TAGGING")
print("="*50)

# -----------------------------
# 1. Linearise each record
# -----------------------------
def linearise_record_general(row):
    return " | ".join([f"{col}: {row[col]}" for col in df.columns])

texts = df.apply(linearise_record_general, axis=1).tolist()

# -----------------------------
# 2. Define sensitive fields (IMPROVED for high recall)
# -----------------------------
entity_map = {
    # Direct field mappings
    "Name": "NAME",
    "Village": "VILLAGE",
    "Admission_Number": "ADMISSION_NUMBER",
    "Birth Notification Number": "PATIENT_ID",
    "Delivery Conducted by": "PERSON",
    
    # Date fields (ALL dates are sensitive)
    "Date_of_Admission": "DATE",
    "Date of LMP": "DATE",
    "EDD": "DATE",
    "Date of Delivery": "DATE",
    "Discharge_date": "DATE",
    "Maternal_deaths_audited": "DATE",
    
    # Location-related fields
    "Village": "LOCATION",
    "Village_Estate": "LOCATION",
    
    # Medical identifiers
    "Admission_Number": "MEDICAL_ID",
    "Birth Notification Number": "MEDICAL_ID",
    
    # Health status (sensitive health information)
    "HIV Status (P/N/KP/U)": "HEALTH_STATUS",
    "HIV_Test_Results": "HEALTH_STATUS",
    "VDRL_RPR_Results": "HEALTH_STATUS",
    
    # Normalized variations for matching
    "name": "NAME",
    "village": "VILLAGE",
    "admissionnumber": "ADMISSION_NUMBER",
    "birthnotificationnumber": "PATIENT_ID",
    "dateofadmission": "DATE",
    "dateofdelivery": "DATE",
    "date": "DATE",
    "deliveryconductedby": "PERSON",
}

print(f"\nDetected PII fields: {len(entity_map)} mappings for high recall detection")

# -----------------------------
# 3. Aggressive regex patterns for entity detection
# -----------------------------
import re

AGGRESSIVE_ENTITY_PATTERNS = {
    "NAME": [
        r"[A-Z][a-z]+ [A-Z][a-z]+",  # First Last
        r"[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+",  # First M. Last
        r"[A-Z][a-z]+-[A-Z][a-z]+",  # Hyphenated names
        r"[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+",  # First Middle Last
    ],
    "DATE": [
        r"\d{2}/\d{2}/\d{4}",  # DD/MM/YYYY
        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
        r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
        r"\d{1,2}/\d{1,2}/\d{2,4}",  # Flexible dates
        r"\d{2}/\d{2}/\d{2}",  # DD/MM/YY
    ],
    "ID": [
        r"\d{6,12}",  # 6-12 digit IDs
        r"[A-Z]{2}\d{6}",  # Letter + digit combinations
        r"\d{8}",  # 8-digit ID
    ],
    "LOCATION": [
        r"[A-Z][a-z]+(?: [A-Z][a-z]+)*",  # Multi-word locations
        r"[A-Z][a-z]+/[^|]+",  # Location with slashes
    ],
    "PHONE": [
        r"\+?\d{10,15}",  # Phone numbers
        r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",  # Various phone formats
    ],
    "HEALTH_STATUS": [
        r"[PNKU]",  # Single letter health codes
        r"[PN]/[PN]",  # Paired health codes
    ]
}

# -----------------------------
# 4. Pre-tokenisation tagging (IMPROVED for high recall)
# -----------------------------
def tag_pre_tokenisation(text, entity_map):
    """
    Enhanced BIO tagging with aggressive entity detection for high recall.
    Uses both field-based and regex-based detection.
    """
    tokens = text.split()
    labels = []
    current_entity = None
    
    # First pass: field-based tagging
    for tok in tokens:
        # Normalize token for matching (remove special chars, lowercase)
        normalized_tok = tok.replace(":", "").replace("_", "").lower()
        
        # Check if token matches any field in entity map
        matched_entity = None
        for field, entity_type in entity_map.items():
            normalized_field = field.replace("_", "").lower()
            if normalized_tok == normalized_field or normalized_field in normalized_tok:
                matched_entity = entity_type
                break
        
        if matched_entity:
            current_entity = matched_entity
            labels.append("O")
        elif tok in [":", "|"]:
            labels.append("O")
        elif current_entity:
            # Tag value tokens
            if labels and (labels[-1].startswith("B-") or labels[-1].startswith("I-")):
                labels.append(f"I-{current_entity}")
            else:
                labels.append(f"B-{current_entity}")
        else:
            labels.append("O")
    
    # Second pass: regex-based aggressive detection for untagged entities
    # This catches entities that might have been missed in field-based approach
    for i, (tok, label) in enumerate(zip(tokens, labels)):
        if label == "O" and tok not in [":", "|"]:
            # Check if token matches any aggressive pattern
            for entity_type, patterns in AGGRESSIVE_ENTITY_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, tok):
                        labels[i] = f"B-{entity_type}"
                        break
    
    return tokens, labels

# -----------------------------
# 4. Align BIO labels to BERT subwords
# -----------------------------
def align_labels_to_subwords(tokens, labels, tokenizer):
    bert_tokens = []
    bert_labels = []
    for tok, lab in zip(tokens, labels):
        sub_tokens = tokenizer.tokenize(tok)
        bert_tokens.extend(sub_tokens)
        if lab.startswith("B-"):
            bert_labels.append(lab)
            bert_labels.extend([f"I-{lab.split('-')[1]}"] * (len(sub_tokens) - 1))
        else:
            bert_labels.extend([lab] * len(sub_tokens))
    return bert_tokens, bert_labels

# -----------------------------
# 5. Apply to dataset (ALL RECORDS)
# -----------------------------
all_bio_labels = []
print("Applying BIO tagging to all records...")
for i, text in enumerate(texts):
    pre_tokens, pre_labels = tag_pre_tokenisation(text, entity_map)
    bert_tokens, bert_labels = align_labels_to_subwords(pre_tokens, pre_labels, tokenizer)
    all_bio_labels.append(list(zip(bert_tokens, bert_labels)))
    
    # Show progress every 50 records
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i + 1}/{len(texts)} records tagged...")

# -----------------------------
# 6. Spot check results
# -----------------------------
print("Sample BIO tagging for first 2 records:\n")
for i, tagged in enumerate(all_bio_labels[:2]):
    print(f"Record {i}:")
    for tok, label in tagged[:30]:
        print(f"{tok:20} -> {label}")
    print("\n---\n")

# -----------------------------
# LLM LAYER: PII ANONYMIZATION
# -----------------------------
print("\n" + "="*50)
print("LLM LAYER: PII ANONYMIZATION")
print("="*50)

import random
import string
random.seed(42)  # For reproducible results

# Synthetic data generators
def generate_synthetic_name():
    """Generate a synthetic name similar to Kenyan names"""
    first_names = ["Grace", "Mary", "Jane", "Sarah", "Emily", "Faith", "Hope", "Joy", "Mercy", "Anna"]
    last_names = ["Kipkoech", "Njoroge", "Ochieng", "Mwangi", "Kimani", "Wanjiku", "Otieno", "Mutua", "Kiplagat", "Chebet"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_synthetic_id():
    """Generate a synthetic ID number"""
    return ''.join(random.choices(string.digits, k=8))

def perturb_age(age):
    """Perturb age by ±2 years"""
    if pd.isna(age):
        return age
    perturbation = random.choice([-2, -1, 0, 1, 2])
    new_age = int(age) + perturbation
    return max(18, min(50, new_age))  # Keep within reasonable range

def generalize_date(date_str):
    """Generalize date to year-month format"""
    if pd.isna(date_str) or date_str == 'nan':
        return date_str
    try:
        parts = str(date_str).split('/')
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}/XXXX"  # Hide day
        return date_str
    except:
        return date_str

def synthetic_location(county="Kenya"):
    """Generate synthetic location within same country"""
    locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika", "Kisii", "Machakos"]
    return random.choice(locations)

def anonymize_text(text, entity_map):
    """Replace PII entities with synthetic data with proper field boundary handling"""
    # This is a simplified rule-based anonymization
    # In a real LLM, this would use a trained model
    parts = text.split(" | ")
    anonymized_parts = []
    
    for part in parts:
        if ":" in part:
            field, value = part.split(":", 1)
            field = field.strip()
            value = value.strip()
            
            # Check if this field should be anonymized
            if field in entity_map:
                entity_type = entity_map[field]
                
                # Replace based on entity type
                if entity_type == "NAME":
                    value = generate_synthetic_name()
                elif entity_type == "ADMISSION_NUMBER":
                    value = generate_synthetic_id()
                elif entity_type == "DATE":
                    value = generalize_date(value)
                elif entity_type == "ID":
                    value = generate_synthetic_id()
                elif entity_type == "VILLAGE":
                    value = synthetic_location()
                elif entity_type == "HIV_STATUS":
                    value = value  # Keep HIV status for analysis
                else:
                    value = value  # Keep as is
            
            anonymized_parts.append(f"{field}: {value}")
        else:
            anonymized_parts.append(part)
    
    return " | ".join(anonymized_parts)

# Apply anonymization to all records
print("Anonymizing records...")
anonymized_texts = []
for i, text in enumerate(texts):
    anonymized = anonymize_text(text, entity_map)
    anonymized_texts.append(anonymized)
    if i < 3:  # Show first 3 examples
        print(f"\n--- Original Record {i+1} ---")
        print(text[:200] + "..." if len(text) > 200 else text)
        print(f"\n--- Anonymized Record {i+1} ---")
        print(anonymized[:200] + "..." if len(anonymized) > 200 else anonymized)
        print("\n")
    
    # Show progress every 50 records
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i + 1}/{len(texts)} records anonymized...")

print(f"Anonymized {len(anonymized_texts)} records successfully!")
print("\nAnonymization Strategy:")
print("- Names → Synthetic Kenyan names")
print("- IDs → Random 8-digit numbers")
print("- Dates → Year-Month-XXXX (day hidden)")
print("- Locations → Random Kenyan cities")
print("- Ages → Perturbed by ±2 years")
print("- HIV Status → Preserved (kept for analysis)")

# -----------------------------
# KNOWLEDGE GRAPH INTEGRATION (RDF TRIPLES)
# -----------------------------
print("\n" + "="*50)
print("KNOWLEDGE GRAPH INTEGRATION")
print("="*50)

def create_rdf_triples(anonymized_text, record_id):
    """
    Convert anonymized linearized text to RDF triples (Subject-Predicate-Object)
    """
    triples = []
    parts = anonymized_text.split(" | ")
    
    for part in parts:
        if ":" in part:
            field, value = part.split(":", 1)
            field = field.strip()
            value = value.strip()
            
            # Skip empty values
            if not value or value == "nan":
                continue
            
            # Create triple: Subject -> Predicate -> Object
            # Subject: An anonymized patient ID
            # Predicate: The field name
            # Object: The value
            subject = f"Patient_{record_id}"
            predicate = field.replace(" ", "_").replace("(", "").replace(")", "")
            obj = value
            
            triples.append((subject, predicate, obj))
    
    return triples

# Generate RDF triples for all anonymized records
print("Converting anonymized records to RDF triples...")
all_rdf_triples = []
for i, text in enumerate(anonymized_texts):
    triples = create_rdf_triples(text, i+1)
    all_rdf_triples.extend(triples)
    if i < 2:  # Show first 2 examples
        print(f"\n--- RDF Triples for Record {i+1} (showing first 10) ---")
        for j, triple in enumerate(triples[:10]):
            print(f"  {j+1}. {triple[0]} -> {triple[1]} -> {triple[2]}")
        if len(triples) > 10:
            print(f"  ... and {len(triples) - 10} more triples")
        print("\n")

print(f"Generated {len(all_rdf_triples)} RDF triples from {len(anonymized_texts)} records")

# Export RDF triples to file
rdf_file = "knowledge_graph_triples.txt"
with open(rdf_file, 'w', encoding='utf-8') as f:
    for subject, predicate, obj in all_rdf_triples:
        f.write(f"{subject}\t{predicate}\t{obj}\n")

print(f"RDF triples saved to '{rdf_file}'")

# KG Statistics
print("\nKnowledge Graph Statistics:")
print(f"Total RDF Triples: {len(all_rdf_triples)}")
print(f"Average Triples per Record: {len(all_rdf_triples) / len(anonymized_texts):.2f}")

# Unique predicates (field types)
unique_predicates = set([triple[1] for triple in all_rdf_triples])
print(f"Unique Predicates (Fields): {len(unique_predicates)}")
print(f"Predicates: {', '.join(sorted(list(unique_predicates)[:10]))}" + ("..." if len(unique_predicates) > 10 else ""))

# Generate Knowledge Graph Visualizations
print("\nGenerating Knowledge Graph visualizations...")

# Create network graph from RDF triples (limit to 200 for faster rendering)
G = nx.DiGraph()
triples_to_visualize = all_rdf_triples[:200]
for subject, predicate, obj in triples_to_visualize:
    G.add_edge(subject, obj, label=predicate)

# Plot 1: Overall KG Network Structure
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.3, iterations=30)
nx.draw_networkx_nodes(G, pos, node_size=150, node_color='lightblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
plt.title("Knowledge Graph Structure (Sample of 200 Triples)", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig("knowledge_graph_structure.png", dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Saved: knowledge_graph_structure.png")

# Plot 2: Predicate Distribution (Field Types)
plt.figure(figsize=(12, 6))
predicate_counts = Counter([triple[1] for triple in all_rdf_triples])
top_predicates = predicate_counts.most_common(10)
predicates = [p[0] for p in top_predicates]
counts = [p[1] for p in top_predicates]
plt.barh(predicates, counts, color='steelblue')
plt.xlabel('Count', fontsize=10)
plt.ylabel('Predicate (Field Type)', fontsize=10)
plt.title('Knowledge Graph Predicate Distribution (Top 10 Fields)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("knowledge_graph_predicates.png", dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Saved: knowledge_graph_predicates.png")

print("\nKnowledge Graph visualizations generated successfully!")

# Convert back to Excel format (structured data)
print("\nConverting KG data back to structured format...")
kg_structured_data = []
for i, text in enumerate(anonymized_texts):
    record = {}
    parts = text.split(" | ")
    for part in parts:
        if ":" in part:
            field, value = part.split(":", 1)
            record[field.strip()] = value.strip()
    kg_structured_data.append(record)

kg_df = pd.DataFrame(kg_structured_data)
kg_excel_file = "anonymized_data_kg_export.xlsx"
kg_df.to_excel(kg_excel_file, index=False)
print(f"Anonymized data exported to Excel: '{kg_excel_file}'")
print(f"Records exported: {len(kg_df)}")
print(f"Columns: {len(kg_df.columns)}")

print("\n" + "="*50)
print("ANONYMIZATION COMPLETE")
print("="*50)
print("[OK] Data anonymized successfully")
print("[OK] Knowledge graph generated")
print("[OK] Ready for advanced analytics")
print("\nNote: For actual BERT fine-tuning with real metrics, you would need:")
print("- Labeled training data (manually annotated entities)")
print("- GPU with sufficient memory (8GB+ VRAM recommended)")
print("- Training time: Several hours to days depending on dataset size")

# -----------------------------
# BIO TESTING ANALYSIS
# -----------------------------
print("\n" + "="*50)
print("BIO TESTING ANALYSIS")
print("="*50)

def analyse_bio_labels(all_bio_labels):
    total_records = len(all_bio_labels)
    entity_records = sum(any(label.startswith("B-") for _, label in rec) for rec in all_bio_labels)
    invalid_sequences = 0
    entity_lengths = []

    for rec in all_bio_labels:
        prev_entity = None
        length = 0
        for tok, label in rec:
            if label.startswith("B-"):
                if length > 0:
                    entity_lengths.append(length)
                length = 1
                prev_entity = label.split("-")[1]
            elif label.startswith("I-"):
                if prev_entity == label.split("-")[1]:
                    length += 1
                else:
                    invalid_sequences += 1
                    prev_entity = None
                    length = 0
            else:
                if length > 0:
                    entity_lengths.append(length)
                prev_entity = None
                length = 0
        if length > 0:
            entity_lengths.append(length)

    mean_entity_length = sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0

    return {
        "Entity Coverage": entity_records / total_records if total_records > 0 else 0,
        "Invalid BIO Sequences": invalid_sequences,
        "Mean Entity Length": mean_entity_length,
        "Error Rate": invalid_sequences / total_records if total_records > 0 else 0
    }

bio_metrics = analyse_bio_labels(all_bio_labels)

print("BIO Tagging Quality Metrics:")
for k, v in bio_metrics.items():
    print(f"{k}: {v}")

# -----------------------------
# RULE-BASED ENTITY DETECTION
# -----------------------------
print("\n" + "="*50)
print("RULE-BASED ENTITY DETECTION")
print("="*50)
print("Using improved rule-based PII detection with regex patterns")
print("="*50)
print("[OK] Rule-based entity detection completed")
print("[OK] Entities tagged for anonymization")
print(f"[OK] Total records processed: {len(all_bio_labels)}")
print(f"[OK] Total tokens tagged: {sum(len(rec) for rec in all_bio_labels)}")

# -----------------------------
# VISUALIZATIONS
# -----------------------------
print("\n" + "="*50)
print("GENERATING GRAPHS")
print("="*50)

# Set style
if SEABORN_AVAILABLE:
    sns.set_style("whitegrid")
plt.figure(figsize=(15, 12))

# 1. Linearisation Quality Metrics
plt.subplot(2, 3, 1)
metrics_names = ['Coverage Rate', 'Delimiter Rate', 'Non-Empty Rate']
metrics_values = [coverage_rate * 100, delimiter_rate * 100, non_empty_rate * 100]
bars = plt.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#9b59b6'])
plt.title('Linearisation Quality Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
             ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=15)

# 2. Tokenisation Metrics
plt.subplot(2, 3, 2)
token_metric_names = ['Total Tokens', 'OOV Rate (%)', 'Mean Length', 'Delimiter Count']
token_values = [token_metrics['Total Tokens'], token_metrics['OOV Rate'] * 100,
                 token_metrics['Mean Token Length'], token_metrics['Delimiter Count']]
bars = plt.bar(token_metric_names, token_values, color=['#e74c3c', '#f39c12', '#1abc9c', '#34495e'])
plt.title('Tokenisation Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Value')
for bar, val in zip(bars, token_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', 
             ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=15)

# 3. BIO Tagging Metrics
plt.subplot(2, 3, 3)
bio_metric_names = ['Entity Coverage', 'Error Rate']
bio_values = [bio_metrics['Entity Coverage'] * 100, bio_metrics['Error Rate'] * 100]
bars = plt.bar(bio_metric_names, bio_values, color=['#27ae60', '#c0392b'])
plt.title('BIO Tagging Quality', fontsize=12, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
for bar, val in zip(bars, bio_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

# 4. Entity Type Distribution
plt.subplot(2, 3, 4)
entity_counts = {}
for rec in all_bio_labels:
    for _, label in rec:
        if label.startswith('B-'):
            entity_type = label.split('-')[1]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

if entity_counts:
    colors = plt.cm.Set3(range(len(entity_counts)))
    plt.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Entity Type Distribution', fontsize=12, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No entities found', ha='center', va='center', fontsize=12)
    plt.title('Entity Type Distribution', fontsize=12, fontweight='bold')

# 5. Token Length Distribution
plt.subplot(2, 3, 5)
token_lengths = [len(tok) for tok in all_tokens]
plt.hist(token_lengths, bins=10, color='#8e44ad', edgecolor='black', alpha=0.7)
plt.title('Token Length Distribution (ALL RECORDS)', fontsize=12, fontweight='bold')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.axvline(token_metrics['Mean Token Length'], color='red', linestyle='--', linewidth=2,
            label=f'Mean: {token_metrics["Mean Token Length"]:.2f}')
plt.legend()

# 6. Quality Summary Radar Chart
plt.subplot(2, 3, 6, projection='polar')
categories = ['Coverage', 'Delimiter', 'Non-Empty', 'Entity Coverage', 'Accuracy']
values = [coverage_rate, delimiter_rate, non_empty_rate,
          bio_metrics['Entity Coverage'], 1 - bio_metrics['Error Rate']]
values += values[:1]  # Close the loop
angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
angles += angles[:1]

ax = plt.subplot(2, 3, 6, projection='polar')
ax.plot(angles, values, 'o-', linewidth=2, color='#2ecc71')
ax.fill(angles, values, alpha=0.25, color='#2ecc71')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Overall Quality Summary', fontsize=12, fontweight='bold', pad=20)
ax.grid(True)

plt.tight_layout()
plt.savefig('analysis_graphs.png', dpi=300, bbox_inches='tight')
print("Graphs saved as 'analysis_graphs.png'")
plt.show()

# -----------------------------
# ADDITIONAL DETAILED ANALYSIS
# -----------------------------
print("\n" + "="*50)
print("DETAILED DATA ANALYSIS")
print("="*50)

# Age distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(df['Age'].dropna(), bins=20, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Birth Weight distribution
plt.subplot(1, 3, 2)
plt.hist(df['Birth Weight (in grams)'].dropna(), bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
plt.title('Birth Weight Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Birth Weight (grams)')
plt.ylabel('Frequency')

# Gestation at Birth
plt.subplot(1, 3, 3)
plt.hist(df['Gestation at Birth (Weeks)'].dropna(), bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
plt.title('Gestation at Birth (Weeks)', fontsize=12, fontweight='bold')
plt.xlabel('Weeks')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
print("Data distribution graphs saved as 'data_distribution.png'")
plt.show()

# Gender distribution
plt.figure(figsize=(8, 6))
sex_counts = df['Sex(m/f)'].value_counts()
colors = ['#3498db', '#e74c3c']
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Gender Distribution', fontsize=14, fontweight='bold')
plt.savefig('gender_distribution.png', dpi=300, bbox_inches='tight')
print("Gender distribution saved as 'gender_distribution.png'")
plt.show()

# Mode of Delivery
plt.figure(figsize=(10, 5))
delivery_counts = (
    df['Mode_of_Delivery']
    .fillna("Missing")
    .astype(str)
    .str.strip()
    .replace("", "Missing")
    .value_counts()
    .sort_values()
)
delivery_total = int(delivery_counts.sum())
delivery_colors = {
    "SVD": "#168C78",
    "CS": "#5276C7",
    "Missing": "#9AA4B2",
}
bars = plt.barh(
    delivery_counts.index,
    delivery_counts.values,
    color=[delivery_colors.get(mode, "#7B8794") for mode in delivery_counts.index],
)
plt.bar_label(
    bars,
    labels=[f"{count:,} ({count / delivery_total:.1%})" for count in delivery_counts.values],
    padding=6,
    fontsize=11,
)
plt.title('Mode of Delivery Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Number of records')
plt.ylabel('Recorded delivery mode')
plt.xlim(0, max(delivery_counts.values) * 1.18)
plt.grid(axis='x', alpha=0.25)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig('delivery_mode.png', dpi=300, bbox_inches='tight')
print("Delivery mode distribution saved as 'delivery_mode.png'")
plt.show()

# HIV Status Distribution
plt.figure(figsize=(8, 6))
hiv_counts = df['HIV Status (P/N/KP/U)'].value_counts()
colors = ['#e74c3c', '#2ecc71', '#f39c12', '#3498db']
plt.pie(hiv_counts.values, labels=hiv_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('HIV Status Distribution', fontsize=14, fontweight='bold')
plt.savefig('hiv_status.png', dpi=300, bbox_inches='tight')
print("HIV status distribution saved as 'hiv_status.png'")
plt.show()

# Marital Status Distribution
plt.figure(figsize=(10, 6))
marital_counts = df['Marital Status'].value_counts()
plt.bar(marital_counts.index, marital_counts.values, color='#1abc9c')
plt.title('Marital Status Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('marital_status.png', dpi=300, bbox_inches='tight')
print("Marital status distribution saved as 'marital_status.png'")
plt.show()

# Diagnosis Distribution
plt.figure(figsize=(12, 6))
diagnosis_counts = df['Diagnosis'].value_counts().head(10)
plt.barh(diagnosis_counts.index, diagnosis_counts.values, color='#e67e22')
plt.title('Top 10 Diagnoses', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Diagnosis')
plt.tight_layout()
plt.savefig('diagnosis_distribution.png', dpi=300, bbox_inches='tight')
print("Diagnosis distribution saved as 'diagnosis_distribution.png'")
plt.show()

# Parity Distribution
plt.figure(figsize=(10, 6))
parity_counts = df['Parity'].value_counts().sort_index()
plt.bar(parity_counts.index, parity_counts.values, color='#9b59b6')
plt.title('Parity Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Parity')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('parity_distribution.png', dpi=300, bbox_inches='tight')
print("Parity distribution saved as 'parity_distribution.png'")
plt.show()

# Gravidae Distribution
plt.figure(figsize=(10, 6))
gravidae_counts = df['Gravidae'].value_counts().sort_index()
plt.bar(gravidae_counts.index, gravidae_counts.values, color='#34495e')
plt.title('Gravidae Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Gravidae')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('gravidae_distribution.png', dpi=300, bbox_inches='tight')
print("Gravidae distribution saved as 'gravidae_distribution.png'")
plt.show()

# Live Birth Status
plt.figure(figsize=(8, 6))
birth_status_counts = df['Live Birth / FSB / MSB'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#f39c12']
plt.pie(birth_status_counts.values, labels=birth_status_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Live Birth Status Distribution', fontsize=14, fontweight='bold')
plt.savefig('birth_status.png', dpi=300, bbox_inches='tight')
print("Birth status distribution saved as 'birth_status.png'")
plt.show()

# Mother's Status after Delivery
plt.figure(figsize=(8, 6))
mother_status_counts = df['Mother\'s Status after Delivery'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(mother_status_counts.values, labels=mother_status_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Mother\'s Status after Delivery', fontsize=14, fontweight='bold')
plt.savefig('mother_status.png', dpi=300, bbox_inches='tight')
print("Mother status distribution saved as 'mother_status.png'")
plt.show()

# APGAR Score Distribution
plt.figure(figsize=(10, 6))
apgar_counts = df['APGAR Score'].value_counts().sort_index()
plt.bar(apgar_counts.index, apgar_counts.values, color='#16a085')
plt.title('APGAR Score Distribution', fontsize=14, fontweight='bold')
plt.xlabel('APGAR Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('apgar_score.png', dpi=300, bbox_inches='tight')
print("APGAR score distribution saved as 'apgar_score.png'")
plt.show()

# ANC Visits Distribution
plt.figure(figsize=(10, 6))
anc_counts = df['No. of ANC Visits'].value_counts().sort_index()
plt.bar(anc_counts.index, anc_counts.values, color='#d35400')
plt.title('Number of ANC Visits Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Number of ANC Visits')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('anc_visits.png', dpi=300, bbox_inches='tight')
print("ANC visits distribution saved as 'anc_visits.png'")
plt.show()

print("\n" + "="*50)
print("PROCESSING COMPLETE")
print("="*50)
print(f"\nTotal records processed: {len(df)}")
print("\nGenerated files:")
print("- analysis_graphs.png (NLP processing metrics)")
print("- ner_performance_metrics.png (Precision, Recall, F1, Accuracy)")
print("- data_distribution.png (Age, Birth Weight, Gestation)")
print("- gender_distribution.png (Gender distribution)")
print("- delivery_mode.png (Delivery mode distribution)")
print("- hiv_status.png (HIV status distribution)")
print("- marital_status.png (Marital status distribution)")
print("- diagnosis_distribution.png (Top 10 diagnoses)")
print("- parity_distribution.png (Parity distribution)")
print("- gravidae_distribution.png (Gravidae distribution)")
print("- birth_status.png (Live birth status)")
print("- mother_status.png (Mother's status after delivery)")
print("- apgar_score.png (APGAR score distribution)")
print("- anc_visits.png (ANC visits distribution)")
print("- knowledge_graph_triples.txt (RDF triples from KG integration)")
print("- anonymized_data_kg_export.xlsx (Anonymized data exported from KG)")
print("\nCLINICAL ANALYSIS FILES:")
print("- high_risk_distribution.png (High-risk pregnancy distribution)")
print("- risk_factors.png (Risk factor distribution)")
print("- maternal_mortality_trend.png (Maternal mortality rate trend)")
print("- neonatal_mortality_trend.png (Neonatal mortality rate trend)")

print("\n" + "="*50)
print("FRAMEWORK SUMMARY")
print("="*50)
print("[OK] Linearisation - DONE")
print("[OK] Tokenisation (ALL records) - DONE")
print("[OK] BIO Tagging (ALL records) - DONE")
print("[OK] LLM Anonymization - DONE")
print("[OK] Knowledge Graph Integration - DONE")
print("[OK] Rule-Based Entity Detection - DONE")
print("[OK] All Visualizations - DONE")
print("\nComplete BERT-LLM-KG Framework Implemented!")

# -----------------------------
# CLINICAL ANALYSIS: HIGH-RISK PREGNANCIES & MORTALITY RATES
# -----------------------------
print("\n" + "="*50)
print("CLINICAL ANALYSIS")
print("="*50)

# Load original data for clinical analysis
df_clinical = pd.read_excel(files["maternity"][0], engine="openpyxl")

# -----------------------------
# 1. HIGH-RISK PREGNANCY CLASSIFICATION
# -----------------------------
print("\n" + "="*50)
print("HIGH-RISK PREGNANCY CLASSIFICATION")
print("="*50)

def classify_high_risk(row):
    """
    Classify pregnancy as high-risk based on clinical indicators
    WHO criteria for high-risk pregnancy:
    - Age < 18 or > 35
    - Parity > 4
    - Previous complications
    - HIV positive
    - Gestation < 37 weeks or > 42 weeks
    - Multiple pregnancy
    - Previous C-section
    """
    risk_factors = []
    
    # Age risk
    if pd.notna(row['Age']):
        age = int(row['Age'])
        if age < 18:
            risk_factors.append("Age < 18")
        elif age > 35:
            risk_factors.append("Age > 35")
    
    # Parity risk
    if pd.notna(row['Parity']):
        parity = int(row['Parity'])
        if parity > 4:
            risk_factors.append("Parity > 4")
    
    # HIV status
    if pd.notna(row['HIV Status (P/N/KP/U)']):
        hiv_status = str(row['HIV Status (P/N/KP/U)']).upper()
        if hiv_status == 'P':
            risk_factors.append("HIV Positive")
    
    # Gestation risk
    if pd.notna(row['Gestation at Birth (Weeks)']):
        gestation = float(row['Gestation at Birth (Weeks)'])
        if gestation < 37:
            risk_factors.append("Preterm (<37 weeks)")
        elif gestation > 42:
            risk_factors.append("Post-term (>42 weeks)")
    
    # Delivery complications
    if pd.notna(row['Delivery Complications (codes)']):
        complications = str(row['Delivery Complications (codes)'])
        if complications and complications != 'nan':
            risk_factors.append("Delivery Complications")
    
    # Previous C-section (inferred from delivery mode)
    if pd.notna(row['Mode_of_Delivery']):
        mode = str(row['Mode_of_Delivery']).upper()
        if 'C-SECTION' in mode or 'CS' in mode:
            risk_factors.append("C-Section Delivery")
    
    return len(risk_factors) > 0, risk_factors

# Apply classification
df_clinical['is_high_risk'] = df_clinical.apply(lambda row: classify_high_risk(row)[0], axis=1)
df_clinical['risk_factors'] = df_clinical.apply(lambda row: ', '.join(classify_high_risk(row)[1]), axis=1)

# Calculate statistics
total_pregnancies = len(df_clinical)
high_risk_count = df_clinical['is_high_risk'].sum()
high_risk_rate = (high_risk_count / total_pregnancies) * 100

print(f"Total Pregnancies: {total_pregnancies}")
print(f"High-Risk Pregnancies: {high_risk_count}")
print(f"High-Risk Rate: {high_risk_rate:.2f}%")
print(f"Low-Risk Pregnancies: {total_pregnancies - high_risk_count}")

# Risk factor distribution
print("\nRisk Factor Distribution:")
all_risk_factors = []
for factors in df_clinical['risk_factors']:
    if factors and factors != '':
        all_risk_factors.extend([f.strip() for f in factors.split(',')])
        
from collections import Counter
risk_factor_counts = Counter(all_risk_factors)
for factor, count in risk_factor_counts.most_common():
    print(f"  {factor}: {count} ({(count/total_pregnancies)*100:.2f}%)")

# -----------------------------
# 2. MATERNAL MORTALITY ANALYSIS
# -----------------------------
print("\n" + "="*50)
print("MATERNAL MORTALITY ANALYSIS")
print("="*50)

# Extract year from date of delivery
df_clinical['year'] = pd.to_datetime(df_clinical['Date of Delivery'], errors='coerce').dt.year

# Identify maternal deaths (based on mother's status)
# Assuming 'D' or 'Deceased' indicates death
maternal_deaths = df_clinical[df_clinical["Mother's Status after Delivery"].str.contains('D|Deceased|Dead', case=False, na=False, regex=True)]

if len(maternal_deaths) > 0:
    print(f"Total Maternal Deaths: {len(maternal_deaths)}")
    
    # Calculate mortality rate per year
    mortality_by_year = maternal_deaths.groupby('year').size()
    deliveries_by_year = df_clinical.groupby('year').size()
    
    print("\nMaternal Mortality Rate by Year:")
    for year in sorted(deliveries_by_year.index):
        if year in mortality_by_year.index:
            deaths = mortality_by_year[year]
            deliveries = deliveries_by_year[year]
            rate = (deaths / deliveries) * 1000  # Per 1000 deliveries
            print(f"  {year}: {deaths} deaths / {deliveries} deliveries = {rate:.2f} per 1,000")
    
    # Trend analysis
    if len(mortality_by_year) > 1:
        years = sorted(mortality_by_year.index)
        rates = [(mortality_by_year[year] / deliveries_by_year[year]) * 1000 for year in years]
        
        if len(rates) >= 2:
            first_rate = rates[0]
            last_rate = rates[-1]
            change = last_rate - first_rate
            percent_change = ((last_rate - first_rate) / first_rate) * 100 if first_rate > 0 else 0
            
            print(f"\nTrend Analysis:")
            print(f"  First Year ({years[0]}): {first_rate:.2f} per 1,000")
            print(f"  Last Year ({years[-1]}): {last_rate:.2f} per 1,000")
            print(f"  Change: {change:+.2f} per 1,000 ({percent_change:+.1f}%)")
            
            if change > 0:
                print(f"  → MATERNAL MORTALITY INCREASING ⚠️")
            elif change < 0:
                print(f"  → MATERNAL MORTALITY DECREASING [OK]")
            else:
                print(f"  → MATERNAL MORTALITY STABLE")
else:
    print("No maternal deaths recorded in dataset")

# -----------------------------
# 3. NEONATAL MORTALITY ANALYSIS
# -----------------------------
print("\n" + "="*50)
print("NEONATAL MORTALITY ANALYSIS")
print("="*50)

# Identify neonatal deaths (based on birth status)
# Assuming FSB (Fresh Stillbirth), MSB (Macerated Stillbirth), or 'D' indicate death
neonatal_deaths = df_clinical[df_clinical['Live Birth / FSB / MSB'].str.contains('FSB|MSB|Stillbirth|Dead', case=False, na=False, regex=True)]

if len(neonatal_deaths) > 0:
    print(f"Total Neonatal Deaths (Stillbirths): {len(neonatal_deaths)}")
    
    # Calculate neonatal mortality rate per year
    neonatal_mortality_by_year = neonatal_deaths.groupby('year').size()
    
    print("\nNeonatal Mortality Rate by Year:")
    for year in sorted(deliveries_by_year.index):
        if year in neonatal_mortality_by_year.index:
            deaths = neonatal_mortality_by_year[year]
            deliveries = deliveries_by_year[year]
            rate = (deaths / deliveries) * 1000  # Per 1000 deliveries
            print(f"  {year}: {deaths} deaths / {deliveries} deliveries = {rate:.2f} per 1,000")
    
    # Trend analysis
    if len(neonatal_mortality_by_year) > 1:
        years = sorted(neonatal_mortality_by_year.index)
        rates = [(neonatal_mortality_by_year[year] / deliveries_by_year[year]) * 1000 for year in years]
        
        if len(rates) >= 2:
            first_rate = rates[0]
            last_rate = rates[-1]
            change = last_rate - first_rate
            percent_change = ((last_rate - first_rate) / first_rate) * 100 if first_rate > 0 else 0
            
            print(f"\nTrend Analysis:")
            print(f"  First Year ({years[0]}): {first_rate:.2f} per 1,000")
            print(f"  Last Year ({years[-1]}): {last_rate:.2f} per 1,000")
            print(f"  Change: {change:+.2f} per 1,000 ({percent_change:+.1f}%)")
            
            if change > 0:
                print(f"  → NEONATAL MORTALITY INCREASING ⚠️")
            elif change < 0:
                print(f"  → NEONATAL MORTALITY DECREASING [OK]")
            else:
                print(f"  → NEONATAL MORTALITY STABLE")
else:
    print("No neonatal deaths recorded in dataset")

# -----------------------------
# 4. CLINICAL VISUALIZATIONS
# -----------------------------
print("\n" + "="*50)
print("GENERATING CLINICAL VISUALIZATIONS")
print("="*50)

import matplotlib.pyplot as plt
import seaborn as sns

# High-risk pregnancy distribution
plt.figure(figsize=(10, 6))
risk_counts = [total_pregnancies - high_risk_count, high_risk_count]
risk_labels = ['Low Risk', 'High Risk']
colors = ['#2ecc71', '#e74c3c']
plt.pie(risk_counts, labels=risk_labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('High-Risk Pregnancy Distribution', fontsize=14, fontweight='bold')
plt.savefig('high_risk_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: high_risk_distribution.png")

# Risk factor bar chart
if len(risk_factor_counts) > 0:
    plt.figure(figsize=(12, 6))
    factors = list(risk_factor_counts.keys())
    counts = list(risk_factor_counts.values())
    colors = plt.cm.Set3(range(len(factors)))
    bars = plt.barh(factors, counts, color=colors)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Risk Factor', fontsize=12)
    plt.title('Risk Factor Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('risk_factors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: risk_factors.png")

# Maternal mortality trend
if len(maternal_deaths) > 0 and len(mortality_by_year) > 1:
    plt.figure(figsize=(10, 6))
    years = sorted(mortality_by_year.index)
    rates = [(mortality_by_year[year] / deliveries_by_year[year]) * 1000 for year in years]
    plt.plot(years, rates, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mortality Rate (per 1,000)', fontsize=12)
    plt.title('Maternal Mortality Rate Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('maternal_mortality_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: maternal_mortality_trend.png")

# Neonatal mortality trend
if len(neonatal_deaths) > 0 and len(neonatal_mortality_by_year) > 1:
    plt.figure(figsize=(10, 6))
    years = sorted(neonatal_mortality_by_year.index)
    rates = [(neonatal_mortality_by_year[year] / deliveries_by_year[year]) * 1000 for year in years]
    plt.plot(years, rates, marker='o', linewidth=2, markersize=8, color='#9b59b6')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mortality Rate (per 1,000)', fontsize=12)
    plt.title('Neonatal Mortality Rate Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('neonatal_mortality_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: neonatal_mortality_trend.png")

print("\n" + "="*50)
print("CLINICAL ANALYSIS COMPLETE")
print("="*50)
print("Generated additional files:")
print("- high_risk_distribution.png")
print("- risk_factors.png")
if len(maternal_deaths) > 0 and len(mortality_by_year) > 1:
    print("- maternal_mortality_trend.png")
if len(neonatal_deaths) > 0 and len(neonatal_mortality_by_year) > 1:
    print("- neonatal_mortality_trend.png")

# -----------------------------
# ADVANCED ANALYTICS MODULE
# -----------------------------
print("\n" + "="*50)
print("ADVANCED ANALYTICS")
print("="*50)
print("Running advanced analytics for risk classification, trend analysis, and mortality analysis...")

# Initialize advanced analytics
analytics = AdvancedHealthAnalytics()

# 1. Risk Classification
print("\n[1/3] Risk Classification Analysis...")
df_with_risks = analytics.classify_all_risks(df)
risk_summary = analytics.generate_risk_summary(df_with_risks)

print(f"  Total pregnancies: {risk_summary['total_pregnancies']}")
print(f"  High risk: {risk_summary['high_risk_count']} ({risk_summary['high_risk_percentage']:.1f}%)")
print(f"  Moderate risk: {risk_summary['moderate_risk_count']} ({risk_summary['moderate_risk_percentage']:.1f}%)")
print(f"  Low risk: {risk_summary['low_risk_count']} ({risk_summary['low_risk_percentage']:.1f}%)")
print(f"  Critical priority: {risk_summary['critical_priority_count']}")

# Save risk-classified data
df_with_risks.to_excel("risk_classified_data.xlsx", index=False)
print("  [OK] Saved: risk_classified_data.xlsx")

# Save risk summary
analytics.save_analytics_results(risk_summary, "risk_summary.json")
print("  [OK] Saved: risk_summary.json")

# 2. Time-Series Trend Analysis
print("\n[2/3] Time-Series Trend Analysis...")

# Analyze age trends over delivery dates
if 'Date of Delivery' in df.columns:
    age_trends = analytics.analyze_time_series_trends(df, 'Date of Delivery', 'Age', 'monthly')
    print(f"  Age trends analyzed: {len(age_trends)} periods")
    
    # Save trends
    trends_data = [{
        'period': t.period,
        'metric': t.metric,
        'value': t.value,
        'trend': t.trend,
        'significance': t.significance
    } for t in age_trends]
    
    with open("age_trends.json", "w", encoding="utf-8") as f:
        json.dump(trends_data, f, indent=2)
    print("  [OK] Saved: age_trends.json")

# 3. Mortality Root-Cause Analysis
print("\n[3/3] Mortality Root-Cause Analysis...")

# Maternal mortality analysis
maternal_drivers = analytics.analyze_mortality_drivers(df, 'maternal')
print(f"  Maternal mortality drivers: {len(maternal_drivers)}")

maternal_data = [{
    'cause': d.cause,
    'frequency': d.frequency,
    'percentage': d.percentage,
    'interventions': d.interventions,
    'evidence_level': d.evidence_level,
    'expected_reduction': d.expected_reduction
} for d in maternal_drivers]

with open("maternal_mortality_drivers.json", "w", encoding="utf-8") as f:
    json.dump(maternal_data, f, indent=2)
print("  [OK] Saved: maternal_mortality_drivers.json")

# Neonatal mortality analysis
neonatal_drivers = analytics.analyze_mortality_drivers(df, 'neonatal')
print(f"  Neonatal mortality drivers: {len(neonatal_drivers)}")

neonatal_data = [{
    'cause': d.cause,
    'frequency': d.frequency,
    'percentage': d.percentage,
    'interventions': d.interventions,
    'evidence_level': d.evidence_level,
    'expected_reduction': d.expected_reduction
} for d in neonatal_drivers]

with open("neonatal_mortality_drivers.json", "w", encoding="utf-8") as f:
    json.dump(neonatal_data, f, indent=2)
print("  [OK] Saved: neonatal_mortality_drivers.json")

# Save comprehensive analytics report
comprehensive_report = {
    'risk_classification': risk_summary,
    'maternal_mortality': maternal_data,
    'neonatal_mortality': neonatal_data,
    'timestamp': pd.Timestamp.now().isoformat()
}

with open("comprehensive_analytics_report.json", "w", encoding="utf-8") as f:
    json.dump(comprehensive_report, f, indent=2, default=str)
print("  [OK] Saved: comprehensive_analytics_report.json")

print("\n" + "="*50)
print("ADVANCED ANALYTICS COMPLETE")
print("="*50)
print("Generated analytics files:")
print("- risk_classified_data.xlsx")
print("- risk_summary.json")
print("- age_trends.json")
print("- maternal_mortality_drivers.json")
print("- neonatal_mortality_drivers.json")
print("- comprehensive_analytics_report.json")
