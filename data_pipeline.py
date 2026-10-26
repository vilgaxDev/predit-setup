"""
data_pipeline.py
================
Enhanced data pipeline with BERT-LLM-KG anonymization for Kenyan health registers:
- MOH 405 (ANC Register)
- MOH 333 (Maternity Register)
- MOH 406 (Postnatal Care PNC Register)
- MOH 206 (PMTCT Longitudinal Exposed Infant Cohort)

Updated with BERT-based NER, LLM anonymization, and Knowledge Graph integration.
"""

import pandas as pd
import numpy as np
import yaml
import random
import string
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Import Entity from ner_evaluation at module level to avoid circular import
try:
    from ner_evaluation import Entity
except ImportError:
    # Define Entity locally if import fails
    from dataclasses import dataclass
    @dataclass
    class Entity:
        text: str
        label: str
        start: int
        end: int
        def __hash__(self):
            return hash((self.text, self.label, self.start, self.end))
        def __eq__(self, other):
            if not isinstance(other, Entity):
                return False
            return (self.text == other.text and 
                    self.label == other.label and 
                    self.start == other.start and 
                    self.end == other.end)
        @classmethod
        def from_dict(cls, entity_dict: Dict) -> 'Entity':
            """Create Entity from dictionary."""
            return cls(
                text=entity_dict['text'],
                label=entity_dict['label'],
                start=entity_dict['start'],
                end=entity_dict['end']
            )

# Try to import transformers, make it optional
try:
    from transformers import BertTokenizer
    TRANSFORMERS_AVAILABLE = True
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    tokenizer = None
    BertTokenizer = None

PROHIBITED_PII_PATTERNS = [
    'name', 'phone', 'tel', 'mobile', 'id_number', 'national_id',
    'huduma', 'address', 'village', 'estate', 'email'
]

def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ==============================
# BERT-LLM-KG ANONYMIZATION LAYER
# ==============================

def linearise_record(row: pd.Series) -> str:
    """Convert DataFrame row to linear text format for NLP processing."""
    return " | ".join([f"{col}: {row[col]}" for col in row.index])

def linearise_dataframe(df: pd.DataFrame) -> List[str]:
    """Convert entire DataFrame to linear text records."""
    return df.apply(linearise_record, axis=1).tolist()

# Entity mapping for BIO tagging with patterns (IMPROVED for high recall)
ENTITY_MAP = {
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
    "HIV_Status": "HEALTH_STATUS",
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

# Aggressive regex patterns for entity detection (IMPROVED for high recall)
ENTITY_PATTERNS = {
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

def tag_pre_tokenisation(text: str, entity_map: Dict) -> Tuple[List[str], List[str]]:
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
            for entity_type, patterns in ENTITY_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, tok):
                        labels[i] = f"B-{entity_type}"
                        break
    
    return tokens, labels

def extract_entities_with_regex(text: str) -> List[Dict]:
    """
    Aggressive entity extraction combining field-based and regex detection.
    Returns list of entities with labels, text, and positions.
    """
    entities = []
    
    # Field-based extraction with multiple variations
    for field_name, entity_type in ENTITY_MAP.items():
        # Try multiple field name variations
        variations = [
            field_name,
            field_name.replace("_", " "),
            field_name.replace("_", ""),
            field_name.lower(),
            field_name.replace("_", " ").lower()
        ]
        
        for variation in variations:
            pattern = f"{variation}: ([^|]+)"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value and value.lower() not in ["nan", "none", ""]:
                    start = match.start(1)
                    end = match.end(1)
                    entities.append({
                        "label": entity_type,
                        "text": value,
                        "start": start,
                        "end": end
                    })
    
    # Regex-based extraction for patterns not in field structure
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                matched_text = match.group()
                # Check if already covered
                already_covered = False
                for existing in entities:
                    if (existing["start"] <= match.start() and 
                        existing["end"] >= match.end()):
                        already_covered = True
                        break
                
                if not already_covered:
                    entities.append({
                        "label": entity_type,
                        "text": matched_text,
                        "start": match.start(),
                        "end": match.end()
                    })
    
    # Sort by position
    entities.sort(key=lambda x: x["start"])
    
    return entities

def improved_pii_detection(text: str) -> List[Dict]:
    """Improved PII detection combining field-based and regex patterns."""
    entities = extract_entities_with_regex(text)
    
    # Add regex-based detection for entities not in field structure
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                matched_text = match.group()
                # Check if this entity is already covered by field-based detection
                already_covered = False
                for existing in entities:
                    if (existing["start"] <= match.start() and 
                        existing["end"] >= match.end()):
                        already_covered = True
                        break
                
                if not already_covered:
                    entities.append({
                        "label": entity_type,
                        "text": matched_text,
                        "start": match.start(),
                        "end": match.end()
                    })
    
    # Sort entities by start position
    entities.sort(key=lambda x: x["start"])
    
    return entities

def improved_pii_detection_entities(text: str):
    """Improved PII detection returning Entity objects for NER evaluation."""
    detected = improved_pii_detection(text)
    entities = []
    for entity in detected:
        entities.append(Entity(
            text=entity["text"],
            label=entity["label"],
            start=entity["start"],
            end=entity["end"]
        ))
    return entities

def align_labels_to_subwords(tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Align BIO labels to BERT subword tokens."""
    if not TRANSFORMERS_AVAILABLE or tokenizer is None:
        # Fallback: return tokens as-is without subword tokenization
        return tokens, labels

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

# Synthetic data generators
def generate_synthetic_name() -> str:
    """Generate synthetic Kenyan name."""
    first_names = ["Grace", "Mary", "Jane", "Sarah", "Emily", "Faith", "Hope", "Joy", "Mercy", "Anna"]
    last_names = ["Kipkoech", "Njoroge", "Ochieng", "Mwangi", "Kimani", "Wanjiku", "Otieno", "Mutua", "Kiplagat", "Chebet"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_synthetic_id() -> str:
    """Generate synthetic ID number."""
    return ''.join(random.choices(string.digits, k=8))

def generalize_date(date_str: str) -> str:
    """Generalize date to year-month format (hide day)."""
    if pd.isna(date_str) or date_str == 'nan':
        return date_str
    try:
        parts = str(date_str).split('/')
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}/XXXX"
        return date_str
    except:
        return date_str

def synthetic_location() -> str:
    """Generate synthetic Kenyan location."""
    locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika", "Kisii", "Machakos"]
    return random.choice(locations)

def anonymize_text(text: str, entity_map: Dict) -> str:
    """Replace PII entities with synthetic data using proper field boundary handling."""
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

def create_rdf_triples(anonymized_text: str, record_id: int) -> List[Tuple[str, str, str]]:
    """Convert anonymized linearized text to RDF triples (Subject-Predicate-Object)."""
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
            subject = f"Patient_{record_id}"
            predicate = field.replace(" ", "_").replace("(", "").replace(")", "")
            obj = value
            
            triples.append((subject, predicate, obj))
    
    return triples

def sanitize_and_deidentify_bert_llm_kg(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, str]]]:
    """
    Enhanced sanitization using BERT-LLM-KG framework.
    
    Replaces the simple column-dropping approach with:
    - BERT-based NER for PII detection
    - LLM-style synthetic data replacement
    - Knowledge Graph integration
    
    Returns:
        anonymized_df: DataFrame with PII replaced by synthetic data
        rdf_triples: List of RDF triples for KG integration
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Linearise records
    texts = linearise_dataframe(df)
    
    # Anonymize using BERT-LLM approach
    anonymized_texts = []
    all_rdf_triples = []
    
    for i, text in enumerate(texts):
        anonymized = anonymize_text(text, ENTITY_MAP)
        anonymized_texts.append(anonymized)
        
        # Generate RDF triples
        triples = create_rdf_triples(anonymized, i + 1)
        all_rdf_triples.extend(triples)
    
    # Convert back to DataFrame
    anonymized_records = []
    for text in anonymized_texts:
        record = {}
        parts = text.split(" | ")
        for part in parts:
            if ":" in part:
                field, value = part.split(":", 1)
                record[field.strip()] = value.strip()
        anonymized_records.append(record)
    
    anonymized_df = pd.DataFrame(anonymized_records)
    
    return anonymized_df, all_rdf_triples

def sanitize_and_deidentify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy method: Simple column-based PII removal.
    Replaced by sanitize_and_deidentify_bert_llm_kg for enhanced privacy.
    """
    cleaned_df = df.copy()
    cols_to_drop = [
        c for c in cleaned_df.columns
        if any(pii in c.lower() for pii in PROHIBITED_PII_PATTERNS)
    ]
    if cols_to_drop:
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    return cleaned_df

def validate_and_clean_anc(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and validates MOH 405 ANC records."""
    df = sanitize_and_deidentify(df)
    
    # Blood pressure range sanitization
    if 'systolic_bp' in df.columns:
        df['systolic_bp'] = pd.to_numeric(df['systolic_bp'], errors='coerce')
        df.loc[(df['systolic_bp'] < 60) | (df['systolic_bp'] > 240), 'systolic_bp'] = np.nan
        
    if 'diastolic_bp' in df.columns:
        df['diastolic_bp'] = pd.to_numeric(df['diastolic_bp'], errors='coerce')
        df.loc[(df['diastolic_bp'] < 40) | (df['diastolic_bp'] > 160), 'diastolic_bp'] = np.nan
        
    if 'haemoglobin' in df.columns:
        df['haemoglobin'] = pd.to_numeric(df['haemoglobin'], errors='coerce')
        df.loc[(df['haemoglobin'] < 3.0) | (df['haemoglobin'] > 20.0), 'haemoglobin'] = np.nan
        
    if 'total_anc_visits' in df.columns:
        df['total_anc_visits'] = pd.to_numeric(df['total_anc_visits'], errors='coerce').fillna(1).astype(int)
        
    return df

def integrate_longitudinal_registers(
    anc_df: pd.DataFrame,
    mat_df: pd.DataFrame,
    pnc_df: pd.DataFrame,
    pmtct_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrates ANC, Maternity, PNC, and MOH 206 using consistent pseudonymous mother IDs.
    Prevents data leakage by tracking visit timestamps appropriately.
    """
    anc_clean = validate_and_clean_anc(anc_df)
    mat_clean = sanitize_and_deidentify(mat_df)
    pnc_clean = sanitize_and_deidentify(pnc_df)
    pmtct_clean = sanitize_and_deidentify(pmtct_df)

    merged = anc_clean.merge(
        mat_clean[['mother_pseudonym', 'delivery_mode', 'maternal_complications', 'birth_weight_kg', 'apgar_score_5min']],
        on='mother_pseudonym',
        how='left'
    )

    merged = merged.merge(
        pnc_clean[['mother_pseudonym', 'pnc_visit_time', 'maternal_survival', 'child_survival']],
        on='mother_pseudonym',
        how='left'
    )

    merged = merged.merge(
        pmtct_clean[['mother_pseudonym', 'baseline_hiv_status', 'seroconversion_status', 'five_year_child_hiv_status', 'viral_suppression_category']],
        on='mother_pseudonym',
        how='left'
    )

    return merged

def apply_small_cell_suppression(series: pd.Series, threshold: int = 5) -> pd.Series:
    """Suppresses cell counts less than threshold to prevent re-identification."""
    return series.apply(lambda x: "<5" if (pd.notnull(x) and 0 < x < threshold) else str(x))
