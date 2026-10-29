"""
llm_anonymizer.py
==================
Real LLM integration for PII anonymization using OpenAI API.
Supports multiple LLM providers and prompting strategies.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class LLMConfig:
    """Configuration for LLM-based anonymization."""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"  # gpt-4, gpt-3.5-turbo, claude-3-opus
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    prompting_strategy: str = "few-shot"  # zero-shot, few-shot, chain-of-thought

class LLMAnonymizer:
    """LLM-based PII anonymization with API integration."""
    
    def __init__(self, config: LLMConfig = None):
        """
        Initialize LLM anonymizer.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self.api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            print("⚠️ No API key found. LLM features will use fallback rule-based generation.")
        
        # Try to import OpenAI
        try:
            import openai
            self.openai = openai
            if self.api_key:
                self.openai.api_key = self.api_key
            self.openai_available = True
        except ImportError:
            self.openai = None
            self.openai_available = False
            print("⚠️ OpenAI library not installed. Install with: pip install openai")
    
    def anonymize_text_llm(
        self,
        text: str,
        entity_types: List[str] = None
    ) -> str:
        """
        Anonymize text using LLM.
        
        Args:
            text: Input text with PII
            entity_types: List of entity types to anonymize
            
        Returns:
            Anonymized text
        """
        if not self.openai_available or not self.api_key:
            # Fallback to rule-based
            return self._fallback_anonymization(text)
        
        entity_types = entity_types or ["NAME", "DATE", "ID", "LOCATION", "PHONE"]
        
        # Construct prompt based on strategy
        if self.config.prompting_strategy == "zero-shot":
            prompt = self._construct_zero_shot_prompt(text, entity_types)
        elif self.config.prompting_strategy == "few-shot":
            prompt = self._construct_few_shot_prompt(text, entity_types)
        elif self.config.prompting_strategy == "chain-of-thought":
            prompt = self._construct_cot_prompt(text, entity_types)
        else:
            prompt = self._construct_zero_shot_prompt(text, entity_types)
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a medical data anonymization assistant. Replace personally identifiable information with synthetic data while preserving medical context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            anonymized = response.choices[0].message.content
            return anonymized
            
        except Exception as e:
            print(f"⚠️ LLM API error: {e}")
            print("Falling back to rule-based anonymization")
            return self._fallback_anonymization(text)
    
    def _construct_zero_shot_prompt(self, text: str, entity_types: List[str]) -> str:
        """Construct zero-shot prompt."""
        entity_str = ", ".join(entity_types)
        return f"""Anonymize the following medical text by replacing these PII types with synthetic data: {entity_str}.

Text to anonymize:
{text}

Replace:
- Names with realistic synthetic names
- Dates with generalized dates (hide day if possible)
- IDs with random 8-digit numbers
- Locations with realistic Kenyan cities
- Phone numbers with random phone numbers

Return only the anonymized text without explanation."""
    
    def _construct_few_shot_prompt(self, text: str, entity_types: List[str]) -> str:
        """Construct few-shot prompt with examples."""
        entity_str = ", ".join(entity_types)
        return f"""Anonymize the following medical text by replacing PII with synthetic data: {entity_str}.

Example 1:
Input: "Patient: Grace Rotich, Age: 29, Village: Kapsoit, Date: 21/01/2014, ID: 300001"
Output: "Patient: Mary Njoroge, Age: 27, Village: Nairobi, Date: 21/01/XXXX, ID: 84739201"

Example 2:
Input: "Mother: Jane Ochieng, Phone: 0722123456, Location: Kericho County Hospital"
Output: "Mother: Faith Mutua, Phone: 0711987654, Location: Mombasa County Hospital"

Now anonymize this text:
{text}

Return only the anonymized text without explanation."""
    
    def _construct_cot_prompt(self, text: str, entity_types: List[str]) -> str:
        """Construct chain-of-thought prompt."""
        entity_str = ", ".join(entity_types)
        return f"""Anonymize the following medical text by replacing PII with synthetic data: {entity_str}.

Think step by step:
1. Identify all instances of the entity types
2. For each instance, determine an appropriate synthetic replacement
3. Replace while preserving the medical context
4. Ensure no re-identification is possible

Text to anonymize:
{text}

Provide your reasoning, then the final anonymized text."""
    
    def _fallback_anonymization(self, text: str) -> str:
        """Fallback to rule-based anonymization."""
        from data_pipeline import anonymize_text, ENTITY_MAP
        return anonymize_text(text, ENTITY_MAP)
    
    def batch_anonymize(
        self,
        texts: List[str],
        entity_types: List[str] = None
    ) -> List[str]:
        """
        Anonymize multiple texts.
        
        Args:
            texts: List of input texts
            entity_types: List of entity types to anonymize
            
        Returns:
            List of anonymized texts
        """
        anonymized = []
        for i, text in enumerate(texts):
            print(f"Anonymizing text {i+1}/{len(texts)}...")
            result = self.anonymize_text_llm(text, entity_types)
            anonymized.append(result)
        return anonymized
    
    def set_api_key(self, api_key: str):
        """Set API key for LLM provider."""
        self.api_key = api_key
        if self.openai_available:
            self.openai.api_key = api_key
        print("✓ API key updated")

# Convenience function
def anonymize_with_llm(
    text: str,
    api_key: str = None,
    model: str = "gpt-4",
    strategy: str = "few-shot"
) -> str:
    """
    Convenience function for LLM-based anonymization.
    
    Args:
        text: Input text
        api_key: OpenAI API key
        model: Model to use
        strategy: Prompting strategy
        
    Returns:
        Anonymized text
    """
    config = LLMConfig(
        api_key=api_key,
        model=model,
        prompting_strategy=strategy
    )
    anonymizer = LLMAnonymizer(config)
    return anonymizer.anonymize_text_llm(text)
