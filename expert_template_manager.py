"""
expert_template_manager.py
==========================
Expert template system for structured knowledge extraction from clinical registers.
Supports template-based fact extraction for ANC, PNC, and MOH 204A datasets.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class TemplateField:
    """Represents a field in an expert template."""
    field_name: str
    field_type: str  # TEXT, DATE, NUMBER, CODE, BOOLEAN
    required: bool
    description: str
    validation_pattern: Optional[str] = None
    knowledge_graph_predicate: Optional[str] = None
    ontology_mapping: Optional[str] = None

@dataclass
class ExpertTemplate:
    """Represents an expert-defined template for a specific register."""
    template_id: str
    template_name: str
    register_type: str  # ANC, PNC, MATERNITY, UNDER5
    description: str
    fields: List[TemplateField]
    extraction_rules: Dict[str, Any]
    kg_schema: Dict[str, str]  # Knowledge graph schema mapping

class ExpertTemplateManager:
    """Manages expert templates for clinical register processing."""
    
    def __init__(self, templates_dir: Path = None):
        """
        Initialize template manager.
        
        Args:
            templates_dir: Directory to store template definitions
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "expert_templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        self.templates_file = self.templates_dir / "templates.json"
        self.extractions_file = self.templates_dir / "extracted_facts.jsonl"
        
        # Initialize with default templates if none exist
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default expert templates for each register type."""
        if self.templates_file.exists():
            return
        
        default_templates = {
            "ANC_MOH405": {
                "template_id": "ANC_MOH405",
                "template_name": "ANC Register (MOH 405) Template",
                "register_type": "ANC",
                "description": "Template for extracting facts from Antenatal Care registers",
                "fields": [
                    {
                        "field_name": "ANC_Visit_Date",
                        "field_type": "DATE",
                        "required": True,
                        "description": "Date of ANC visit",
                        "validation_pattern": r"\d{2}/\d{2}/\d{4}",
                        "knowledge_graph_predicate": "has_anc_visit_date",
                        "ontology_mapping": "owl:hasANCVisitDate"
                    },
                    {
                        "field_name": "Gestational_Age",
                        "field_type": "NUMBER",
                        "required": True,
                        "description": "Gestational age in weeks",
                        "validation_pattern": r"\d+",
                        "knowledge_graph_predicate": "has_gestational_age",
                        "ontology_mapping": "owl:hasGestationalAge"
                    },
                    {
                        "field_name": "Blood_Pressure_Systolic",
                        "field_type": "NUMBER",
                        "required": True,
                        "description": "Systolic blood pressure in mmHg",
                        "validation_pattern": r"\d{2,3}",
                        "knowledge_graph_predicate": "has_systolic_bp",
                        "ontology_mapping": "owl:hasSystolicBP"
                    },
                    {
                        "field_name": "Blood_Pressure_Diastolic",
                        "field_type": "NUMBER",
                        "required": True,
                        "description": "Diastolic blood pressure in mmHg",
                        "validation_pattern": r"\d{2,3}",
                        "knowledge_graph_predicate": "has_diastolic_bp",
                        "ontology_mapping": "owl:hasDiastolicBP"
                    },
                    {
                        "field_name": "Hemoglobin_Level",
                        "field_type": "NUMBER",
                        "required": False,
                        "description": "Hemoglobin level in g/dL",
                        "validation_pattern": r"\d+\.\d+",
                        "knowledge_graph_predicate": "has_hemoglobin",
                        "ontology_mapping": "owl:hasHemoglobin"
                    },
                    {
                        "field_name": "HIV_Status",
                        "field_type": "CODE",
                        "required": True,
                        "description": "HIV status (P/N/KP/U)",
                        "validation_pattern": r"[PNKU]",
                        "knowledge_graph_predicate": "has_hiv_status",
                        "ontology_mapping": "owl:hasHIVStatus"
                    },
                    {
                        "field_name": "Tetanus_Toxoid",
                        "field_type": "CODE",
                        "required": False,
                        "description": "Tetanus toxoid vaccination status",
                        "validation_pattern": r"\d",
                        "knowledge_graph_predicate": "has_tetanus_toxoid",
                        "ontology_mapping": "owl:hasTetanusToxoid"
                    },
                    {
                        "field_name": "Intermittent_Prophylactic_Treatment",
                        "field_type": "CODE",
                        "required": False,
                        "description": "IPT for malaria prevention",
                        "validation_pattern": r"[YN]",
                        "knowledge_graph_predicate": "has_ipt",
                        "ontology_mapping": "owl:hasIPT"
                    }
                ],
                "extraction_rules": {
                    "blood_pressure_hypertension": {
                        "condition": "systolic_bp >= 140 or diastolic_bp >= 90",
                        "fact_type": "diagnosis",
                        "fact_value": "hypertension",
                        "knowledge_graph_predicate": "has_diagnosis"
                    },
                    "anemia_detection": {
                        "condition": "hemoglobin < 11.0",
                        "fact_type": "diagnosis",
                        "fact_value": "anemia",
                        "knowledge_graph_predicate": "has_diagnosis"
                    },
                    "hiv_positive_risk": {
                        "condition": "hiv_status == 'P'",
                        "fact_type": "risk_factor",
                        "fact_value": "hiv_positive",
                        "knowledge_graph_predicate": "has_risk_factor"
                    }
                },
                "kg_schema": {
                    "subject_prefix": "ANC_Visit",
                    "namespace": "http://ontology.example.org/anc#",
                    "predicates": {
                        "date": "hasVisitDate",
                        "gestational_age": "hasGestationalAge",
                        "blood_pressure": "hasBloodPressure",
                        "hiv_status": "hasHIVStatus"
                    }
                }
            },
            "PNC_MOH406": {
                "template_id": "PNC_MOH406",
                "template_name": "PNC Register (MOH 406) Template",
                "register_type": "PNC",
                "description": "Template for extracting facts from Postnatal Care registers",
                "fields": [
                    {
                        "field_name": "PNC_Visit_Date",
                        "field_type": "DATE",
                        "required": True,
                        "description": "Date of PNC visit",
                        "validation_pattern": r"\d{2}/\d{2}/\d{4}",
                        "knowledge_graph_predicate": "has_pnc_visit_date",
                        "ontology_mapping": "owl:hasPNCVisitDate"
                    },
                    {
                        "field_name": "Postnatal_Day",
                        "field_type": "NUMBER",
                        "required": True,
                        "description": "Day postpartum",
                        "validation_pattern": r"\d+",
                        "knowledge_graph_predicate": "has_postnatal_day",
                        "ontology_mapping": "owl:hasPostnatalDay"
                    },
                    {
                        "field_name": "Maternal_Survival",
                        "field_type": "CODE",
                        "required": True,
                        "description": "Mother's survival status (A/D)",
                        "validation_pattern": r"[AD]",
                        "knowledge_graph_predicate": "has_maternal_survival",
                        "ontology_mapping": "owl:hasMaternalSurvival"
                    },
                    {
                        "field_name": "Child_Survival",
                        "field_type": "CODE",
                        "required": True,
                        "description": "Child's survival status (A/D)",
                        "validation_pattern": r"[AD]",
                        "knowledge_graph_predicate": "has_child_survival",
                        "ontology_mapping": "owl:hasChildSurvival"
                    },
                    {
                        "field_name": "Breastfeeding_Status",
                        "field_type": "CODE",
                        "required": False,
                        "description": "Breastfeeding initiation status",
                        "validation_pattern": r"[YN]",
                        "knowledge_graph_predicate": "has_breastfeeding_status",
                        "ontology_mapping": "owl:hasBreastfeedingStatus"
                    },
                    {
                        "field_name": "Postpartum_Hemorrhage",
                        "field_type": "BOOLEAN",
                        "required": False,
                        "description": "Postpartum hemorrhage occurrence",
                        "validation_pattern": r"[YN]",
                        "knowledge_graph_predicate": "has_postpartum_hemorrhage",
                        "ontology_mapping": "owl:hasPostpartumHemorrhage"
                    }
                ],
                "extraction_rules": {
                    "maternal_death": {
                        "condition": "maternal_survival == 'D'",
                        "fact_type": "outcome",
                        "fact_value": "maternal_death",
                        "knowledge_graph_predicate": "has_outcome"
                    },
                    "neonatal_death": {
                        "condition": "child_survival == 'D' and postnatal_day <= 28",
                        "fact_type": "outcome",
                        "fact_value": "neonatal_death",
                        "knowledge_graph_predicate": "has_outcome"
                    },
                    "early_pnc_visit": {
                        "condition": "postnatal_day <= 7",
                        "fact_type": "quality_indicator",
                        "fact_value": "early_pnc_visit",
                        "knowledge_graph_predicate": "has_quality_indicator"
                    }
                },
                "kg_schema": {
                    "subject_prefix": "PNC_Visit",
                    "namespace": "http://ontology.example.org/pnc#",
                    "predicates": {
                        "date": "hasVisitDate",
                        "postnatal_day": "hasPostnatalDay",
                        "maternal_survival": "hasMaternalSurvival",
                        "child_survival": "hasChildSurvival"
                    }
                }
            },
            "UNDER5_MOH204A": {
                "template_id": "UNDER5_MOH204A",
                "template_name": "Under-5 Register (MOH 204A) Template",
                "register_type": "UNDER5",
                "description": "Template for extracting facts from Under-5 registers",
                "fields": [
                    {
                        "field_name": "Child_Age_Months",
                        "field_type": "NUMBER",
                        "required": True,
                        "description": "Child's age in months",
                        "validation_pattern": r"\d+",
                        "knowledge_graph_predicate": "has_age_months",
                        "ontology_mapping": "owl:hasAgeMonths"
                    },
                    {
                        "field_name": "Weight_kg",
                        "field_type": "NUMBER",
                        "required": True,
                        "description": "Child's weight in kg",
                        "validation_pattern": r"\d+\.\d+",
                        "knowledge_graph_predicate": "has_weight",
                        "ontology_mapping": "owl:hasWeight"
                    },
                    {
                        "field_name": "Height_cm",
                        "field_type": "NUMBER",
                        "required": False,
                        "description": "Child's height in cm",
                        "validation_pattern": r"\d+",
                        "knowledge_graph_predicate": "has_height",
                        "ontology_mapping": "owl:hasHeight"
                    },
                    {
                        "field_name": "MUAC_mm",
                        "field_type": "NUMBER",
                        "required": False,
                        "description": "Mid-upper arm circumference in mm",
                        "validation_pattern": r"\d+",
                        "knowledge_graph_predicate": "has_muac",
                        "ontology_mapping": "owl:hasMUAC"
                    },
                    {
                        "field_name": "Nutritional_Status",
                        "field_type": "CODE",
                        "required": False,
                        "description": "Nutritional status classification",
                        "validation_pattern": r"\w+",
                        "knowledge_graph_predicate": "has_nutritional_status",
                        "ontology_mapping": "owl:hasNutritionalStatus"
                    },
                    {
                        "field_name": "Vaccination_Status",
                        "field_type": "CODE",
                        "required": True,
                        "description": "Vaccination completion status",
                        "validation_pattern": r"[CY]",
                        "knowledge_graph_predicate": "has_vaccination_status",
                        "ontology_mapping": "owl:hasVaccinationStatus"
                    },
                    {
                        "field_name": "Illness_Diagnosis",
                        "field_type": "TEXT",
                        "required": False,
                        "description": "Primary illness diagnosis",
                        "validation_pattern": r".+",
                        "knowledge_graph_predicate": "has_diagnosis",
                        "ontology_mapping": "owl:hasDiagnosis"
                    }
                ],
                "extraction_rules": {
                    "malnutrition_severe": {
                        "condition": "muac < 115 or weight_for_height < -3",
                        "fact_type": "diagnosis",
                        "fact_value": "severe_acute_malnutrition",
                        "knowledge_graph_predicate": "has_diagnosis"
                    },
                    "malnutrition_moderate": {
                        "condition": "muac >= 115 and muac < 125",
                        "fact_type": "diagnosis",
                        "fact_value": "moderate_acute_malnutrition",
                        "knowledge_graph_predicate": "has_diagnosis"
                    },
                    "underweight": {
                        "condition": "weight_for_age < -2",
                        "fact_type": "diagnosis",
                        "fact_value": "underweight",
                        "knowledge_graph_predicate": "has_diagnosis"
                    },
                    "stunting": {
                        "condition": "height_for_age < -2",
                        "fact_type": "diagnosis",
                        "fact_value": "stunting",
                        "knowledge_graph_predicate": "has_diagnosis"
                    },
                    "vaccination_complete": {
                        "condition": "vaccination_status == 'C'",
                        "fact_type": "quality_indicator",
                        "fact_value": "vaccination_complete",
                        "knowledge_graph_predicate": "has_quality_indicator"
                    }
                },
                "kg_schema": {
                    "subject_prefix": "Under5_Visit",
                    "namespace": "http://ontology.example.org/under5#",
                    "predicates": {
                        "age": "hasAge",
                        "weight": "hasWeight",
                        "height": "hasHeight",
                        "muac": "hasMUAC",
                        "nutritional_status": "hasNutritionalStatus"
                    }
                }
            }
        }
        
        with open(self.templates_file, 'w', encoding='utf-8') as f:
            json.dump(default_templates, f, indent=2)
    
    def load_template(self, template_id: str) -> Optional[Dict]:
        """
        Load a template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template dictionary or None if not found
        """
        if not self.templates_file.exists():
            return None
        
        with open(self.templates_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        return templates.get(template_id)
    
    def get_template_by_register(self, register_type: str) -> Optional[Dict]:
        """
        Get template by register type.
        
        Args:
            register_type: ANC, PNC, MATERNITY, or UNDER5
            
        Returns:
            Template dictionary or None if not found
        """
        if not self.templates_file.exists():
            return None
        
        with open(self.templates_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        for template_id, template in templates.items():
            if template.get('register_type') == register_type:
                return template
        
        return None
    
    def list_templates(self) -> List[Dict]:
        """List all available templates."""
        if not self.templates_file.exists():
            return []
        
        with open(self.templates_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        return [
            {
                'template_id': tid,
                'template_name': t['template_name'],
                'register_type': t['register_type'],
                'description': t['description'],
                'field_count': len(t['fields'])
            }
            for tid, t in templates.items()
        ]
    
    def extract_facts_from_record(
        self,
        record: pd.Series,
        template: Dict,
        record_id: str
    ) -> List[Dict]:
        """
        Extract facts from a clinical record using expert template.
        
        Args:
            record: Pandas Series representing a clinical record
            template: Expert template dictionary
            record_id: Unique identifier for the record
            
        Returns:
            List of extracted facts
        """
        facts = []
        field_map = {field['field_name']: field for field in template['fields']}
        
        # Extract field-level facts
        for field_name, field_def in field_map.items():
            # Try to find matching column in record
            column_name = self._find_matching_column(record, field_name)
            
            if column_name is None and field_def['required']:
                continue  # Skip required fields that don't exist
            
            if column_name is not None:
                value = record.get(column_name)
                
                # Validate value
                if field_def.get('validation_pattern'):
                    pattern = field_def['validation_pattern']
                    if pd.notna(value) and re.match(pattern, str(value)):
                        fact = {
                            'fact_id': f"{record_id}_{field_name}",
                            'record_id': record_id,
                            'fact_type': 'field_value',
                            'field_name': field_name,
                            'value': str(value),
                            'kg_predicate': field_def.get('knowledge_graph_predicate'),
                            'ontology_mapping': field_def.get('ontology_mapping'),
                            'extracted_at': datetime.utcnow().isoformat()
                        }
                        facts.append(fact)
        
        # Apply extraction rules
        extraction_rules = template.get('extraction_rules', {})
        for rule_name, rule in extraction_rules.items():
            if self._evaluate_extraction_rule(record, rule, field_map):
                fact = {
                    'fact_id': f"{record_id}_{rule_name}",
                    'record_id': record_id,
                    'fact_type': rule.get('fact_type', 'derived'),
                    'field_name': rule_name,
                    'value': rule.get('fact_value'),
                    'kg_predicate': rule.get('knowledge_graph_predicate'),
                    'extracted_at': datetime.utcnow().isoformat(),
                    'rule_applied': rule_name
                }
                facts.append(fact)
        
        return facts
    
    def _find_matching_column(self, record: pd.Series, field_name: str) -> Optional[str]:
        """Find matching column name in record for a template field."""
        # Direct match
        if field_name in record.index:
            return field_name
        
        # Case-insensitive match
        for col in record.index:
            if col.lower().replace('_', '').replace(' ', '') == field_name.lower().replace('_', '').replace(' ', ''):
                return col
        
        # Partial match
        for col in record.index:
            if field_name.lower() in col.lower() or col.lower() in field_name.lower():
                return col
        
        return None
    
    def _evaluate_extraction_rule(
        self,
        record: pd.Series,
        rule: Dict,
        field_map: Dict[str, Dict]
    ) -> bool:
        """
        Evaluate an extraction rule against a record.
        
        Args:
            record: Clinical record
            rule: Extraction rule dictionary
            field_map: Mapping of field names to field definitions
            
        Returns:
            True if rule condition is met
        """
        condition = rule.get('condition', '')
        
        # Simple string-based evaluation (in production, use safer eval)
        # Replace field names with actual values
        for field_name in field_map.keys():
            column_name = self._find_matching_column(record, field_name)
            if column_name is not None:
                value = record.get(column_name)
                if pd.notna(value):
                    # Replace field name with value in condition
                    condition = condition.replace(field_name.lower(), str(value))
        
        try:
            # Safe evaluation of simple conditions
            # In production, use a proper expression parser
            return eval(condition, {'__builtins__': {}}, {})
        except:
            return False
    
    def extract_facts_from_dataset(
        self,
        df: pd.DataFrame,
        template_id: str
    ) -> List[Dict]:
        """
        Extract facts from entire dataset using a template.
        
        Args:
            df: DataFrame with clinical records
            template_id: Template identifier
            
        Returns:
            List of all extracted facts
        """
        template = self.load_template(template_id)
        if template is None:
            raise ValueError(f"Template not found: {template_id}")
        
        all_facts = []
        
        for idx, row in df.iterrows():
            record_id = f"{template['register_type']}_record_{idx}"
            facts = self.extract_facts_from_record(row, template, record_id)
            all_facts.extend(facts)
        
        return all_facts
    
    def save_extracted_facts(self, facts: List[Dict]) -> Path:
        """
        Save extracted facts to JSONL file.
        
        Args:
            facts: List of extracted facts
            
        Returns:
            Path to saved file
        """
        with open(self.extractions_file, 'a', encoding='utf-8') as f:
            for fact in facts:
                f.write(json.dumps(fact) + '\n')
        
        return self.extractions_file
    
    def convert_facts_to_kg_triples(
        self,
        facts: List[Dict],
        template: Dict
    ) -> List[Tuple[str, str, str]]:
        """
        Convert extracted facts to RDF triples for knowledge graph.
        
        Args:
            facts: List of extracted facts
            template: Expert template used for extraction
            
        Returns:
            List of RDF triples (subject, predicate, object)
        """
        kg_schema = template.get('kg_schema', {})
        subject_prefix = kg_schema.get('subject_prefix', 'Record')
        namespace = kg_schema.get('namespace', '')
        
        triples = []
        
        for fact in facts:
            if fact.get('kg_predicate'):
                subject = f"{subject_prefix}_{fact['record_id']}"
                predicate = f"{namespace}{fact['kg_predicate']}"
                obj = str(fact['value'])
                
                triples.append((subject, predicate, obj))
        
        return triples
    
    def get_template_statistics(self, template_id: str) -> Dict:
        """
        Get statistics about a template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Dictionary with template statistics
        """
        template = self.load_template(template_id)
        if template is None:
            return {}
        
        return {
            'template_id': template['template_id'],
            'template_name': template['template_name'],
            'register_type': template['register_type'],
            'total_fields': len(template['fields']),
            'required_fields': sum(1 for f in template['fields'] if f['required']),
            'optional_fields': sum(1 for f in template['fields'] if not f['required']),
            'extraction_rules': len(template.get('extraction_rules', {})),
            'kg_predicates': sum(1 for f in template['fields'] if f.get('knowledge_graph_predicate'))
        }
