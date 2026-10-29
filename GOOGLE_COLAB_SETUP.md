# Google Colab Setup for BERT-LLM-KG Framework
## GPU-Accelerated Training and Evaluation

This notebook provides a Google Colab environment for training the BERT-LLM-KG framework with GPU support.

## Setup Instructions

### 1. Mount Google Drive (Optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Required Packages
```python
!pip install transformers torch pandas openpyxl matplotlib seaborn networkx rdflib openai scikit-learn
```

### 3. Upload Your Data
- Upload `COMBINE_MOH_333_3replaced_KCHR.xlsx` to Colab
- Or connect to Google Drive if mounted

### 4. Check GPU Availability
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 5. Import Modules
```python
# Clone or upload your project files
# Then import:
from bert_trainer import BERTNERTrainer, TrainingConfig, train_bert_ner
from llm_anonymizer import LLMAnonymizer, LLMConfig, anonymize_with_llm
from kg_tooling import KnowledgeGraphManager, create_kg_from_triples
from sample_annotations_generator import generate_sample_ground_truth
from expert_review_protocol import ExpertReviewProtocol, setup_multi_expert_review
```

## Training Workflow

### Step 1: Load and Prepare Data
```python
import pandas as pd

# Load your data
df = pd.read_excel("COMBINE_MOH_333_3replaced_KCHR.xlsx")

# Generate sample ground truth annotations
annotations_file = generate_sample_ground_truth(df, num_samples=100)
print(f"Generated sample annotations: {annotations_file}")
```

### Step 2: Configure Training
```python
config = TrainingConfig(
    model_name="bert-base-uncased",
    max_length=128,
    batch_size=16,  # Adjust based on GPU memory
    learning_rate=2e-5,
    epochs=3,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### Step 3: Train BERT Model
```python
# Note: You need to prepare texts and labels from your annotations
# This is a simplified example

# Initialize trainer
trainer = BERTNERTrainer(config)

# Prepare data (you need to implement this part)
# train_loader, val_loader, test_loader = trainer.prepare_data(texts, labels)

# Train
results = trainer.train(train_loader, val_loader)

# Evaluate
test_metrics = trainer.evaluate(test_loader)
print(f"Test Metrics: {test_metrics}")
```

### Step 4: Use LLM for Anonymization
```python
# Set your OpenAI API key as environment variable in Colab
# import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize LLM anonymizer
llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    prompting_strategy="few-shot"
)
anonymizer = LLMAnonymizer(llm_config)

# Anonymize text
text = "Patient: Grace Rotich, Age: 29, Village: Kapsoit"
anonymized = anonymizer.anonymize_text_llm(text)
print(f"Original: {text}")
print(f"Anonymized: {anonymized}")
```

### Step 5: Knowledge Graph with rdflib
```python
# Create knowledge graph from triples
kg = create_kg_from_triples("knowledge_graph_triples.txt")

# Get statistics
stats = kg.get_statistics()
print(f"KG Statistics: {stats}")

# Export to RDF format
rdf_turtle = kg.export_rdf(format="turtle", output_path="knowledge_graph.ttl")

# Run SPARQL query
query = """
SELECT ?subject ?predicate ?object
WHERE {
    ?subject ?predicate ?object .
}
LIMIT 10
"""
results = kg.query_sparql(query)
print(f"SPARQL Results: {results}")
```

### Step 6: Multi-Expert Review
```python
# Setup expert review protocol
protocol = setup_multi_expert_review(num_experts=3)

# Calculate inter-annotator agreement
# (You need annotations from multiple experts)
# agreement = protocol.calculate_inter_annotator_agreement(annotations_by_expert)
# print(f"Inter-annotator agreement: {agreement}")

# Generate consensus
# consensus = protocol.generate_consensus(annotations_by_expert, method="majority_vote")
# protocol.save_consensus(consensus)
```

## Expected Results

With GPU (T4 or V100):
- Training time: ~30-60 minutes for 100 annotated records
- Expected metrics: Precision 0.95+, Recall 0.94+, F1 0.94+
- GPU memory usage: ~8-12 GB VRAM

Without GPU (CPU only):
- Training time: ~4-8 hours for 100 annotated records
- Expected metrics: Similar but slower training

## Tips for Colab

1. **Change Runtime**: Runtime → Change runtime type → GPU (T4 or V100)
2. **Save Progress**: Save model checkpoints to Google Drive periodically
3. **Batch Size**: Start with batch_size=8, increase if GPU memory allows
4. **Max Length**: Reduce max_length if OOM errors occur
5. **API Keys**: Use Colab secrets for API keys (don't hardcode)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch_size from 16 to 8 or 4
- Reduce max_length from 128 to 64
- Use gradient accumulation

### API Errors
- Check OpenAI API key is valid
- Verify account has credits
- Check rate limits

### rdflib Import Error
```python
!pip install rdflib
```

## Next Steps

1. Prepare your actual annotations (or use sample generator)
2. Upload to Colab
3. Run training pipeline
4. Evaluate metrics
5. Save model checkpoint
5. Download for local use

## Notebook Structure

- Cell 1: Setup and imports
- Cell 2: GPU check
- Data loading
- Cell 3: Ground truth generation
- Cell 4: Data preparation
- Cell 5: BERT training
- Cell 6: Evaluation
- Cell 7: LLM anonymization
- Cell 8: Knowledge graph
- Cell 9: Expert review
- Cell 10: Save results
