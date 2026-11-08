"""
All-in-One LoRA Fine-tuning Script for BERT Financial Sentiment Analysis
===========================================================================
This file contains the complete pipeline for fine-tuning BERT with LoRA.
It's structured like a Jupyter notebook but as a single executable Python file.

Usage:
    python all_code.py

Requirements:
    - config.yaml file in the same directory
    - data/combined.jsonl file with financial sentiment data
"""

# =============================================================================
# IMPORTS
# =============================================================================
import json
import yaml
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict


# =============================================================================
# SECTION 1: DATA LOADING UTILITIES
# =============================================================================

def load_jsonl(file_path: Path) -> List[Dict]:
    """
    Load JSONL file and return list of dictionaries.
    
    Each line in the JSONL file should be a valid JSON object with:
    - text: the input text
    - label: the sentiment label (0=negative, 1=neutral, 2=positive)
    - source: optional source identifier
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries from JSONL file
    """
    print(f"Loading data from: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} records")
    return data


def create_datasets(
    data_file: Path,
    text_column: str = "text",
    label_column: str = "label",
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Load data and split into train/validation/test datasets.
    
    This function:
    1. Loads the JSONL file
    2. Extracts text and labels
    3. Performs stratified splitting to maintain label distribution
    4. Creates HuggingFace Dataset objects
    
    Args:
        data_file: Path to JSONL file
        text_column: Name of text column
        label_column: Name of label column
        train_split: Proportion for training set (0.8 = 80%)
        val_split: Proportion for validation set (0.1 = 10%)
        test_split: Proportion for test set (0.1 = 10%)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Load data
    data = load_jsonl(data_file)
    
    # Extract texts and labels
    texts = [item[text_column] for item in data]
    labels = [item[label_column] for item in data]
    
    print(f"Splitting data: {train_split:.0%} train, {val_split:.0%} val, {test_split:.0%} test")
    
    # Split data
    # First split: train vs (val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, 
        test_size=float(val_split + test_split), 
        random_state=int(seed), 
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio = float(val_split) / float(val_split + test_split)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, 
        test_size=float(1 - val_ratio), 
        random_state=int(seed), 
        stratify=temp_labels
    )
    
    # Create HuggingFace Dataset objects
    train_dataset = Dataset.from_dict({text_column: train_texts, label_column: train_labels})
    val_dataset = Dataset.from_dict({text_column: val_texts, label_column: val_labels})
    test_dataset = Dataset.from_dict({text_column: test_texts, label_column: test_labels})
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def preprocess_function(examples, tokenizer, text_column: str, label_column: str, max_length: int):
    """
    Tokenize texts for model input and add labels.
    
    This function:
    1. Tokenizes the input texts
    2. Applies truncation and padding
    3. Adds labels to the tokenized output
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: HuggingFace tokenizer instance
        text_column: Name of text column
        label_column: Name of label column
        max_length: Maximum sequence length (tokens beyond this are truncated)
        
    Returns:
        Dictionary with tokenized inputs and labels
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples[text_column],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # Add labels to the tokenized inputs (required for training)
    tokenized["labels"] = examples[label_column]
    return tokenized


# =============================================================================
# SECTION 2: CONFIGURATION AND METRICS
# =============================================================================

def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.
    
    The config file should contain:
    - model: model name, token, max_length, num_labels
    - data: file paths and column names
    - lora: LoRA hyperparameters
    - training: training hyperparameters
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary with configuration
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.
    
    Calculates:
    - Accuracy: Overall correct predictions
    - F1 Score: Harmonic mean of precision and recall (weighted average)
    - Precision: True positives / (True positives + False positives)
    - Recall: True positives / (True positives + False negatives)
    
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        
    Returns:
        Dictionary with metric names and values
    """
    predictions, labels = eval_pred
    # Convert logits to predicted class (argmax)
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics with weighted average (accounts for class imbalance)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# =============================================================================
# SECTION 3: TAPT (Task-Adaptive Pre-Training) - OPTIONAL
# =============================================================================

def run_tapt_mlm(model_name: str, tokenizer, texts: List[str], tapt_config: dict, use_auth_token: str = None) -> Path:
    """
    Run Task-Adaptive Pre-Training using Masked Language Modeling.
    
    This further adapts a domain-adapted model (DAPT) to your specific task texts.
    
    Args:
        model_name: HuggingFace model name to start from
        tokenizer: Pre-loaded tokenizer
        texts: List of text strings for MLM training
        tapt_config: Configuration dict with TAPT parameters
        use_auth_token: HuggingFace auth token for private models
    
    Returns:
        Path to saved TAPT model
    """
    print("\n" + "="*80)
    print("TAPT STAGE: MASKED LANGUAGE MODELING")
    print("="*80)
    
    tapt_output_dir = Path(tapt_config['output_dir'])
    tapt_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model for masked language modeling
    print(f"Loading base model for MLM: {model_name}")
    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        token=use_auth_token
    )
    
    num_params = sum(p.numel() for p in mlm_model.parameters())
    print(f"Loaded model with {num_params:,} parameters")
    
    # Prepare dataset
    print(f"\nPreparing {len(texts)} texts for MLM training...")
    mlm_dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )
    
    mlm_dataset = mlm_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing for MLM"
    )
    
    # Data collator for MLM (handles masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Mask 15% of tokens
    )
    
    # TAPT training arguments
    mlm_training_args = TrainingArguments(
        output_dir=str(tapt_output_dir),
        num_train_epochs=tapt_config['num_train_epochs'],
        per_device_train_batch_size=tapt_config['per_device_train_batch_size'],
        gradient_accumulation_steps=tapt_config['gradient_accumulation_steps'],
        learning_rate=tapt_config['learning_rate'],
        warmup_ratio=tapt_config['warmup_ratio'],
        weight_decay=tapt_config['weight_decay'],
        fp16=tapt_config.get('fp16', True),
        logging_steps=50,
        save_strategy="epoch",
        seed=tapt_config.get('seed', 42),
        report_to="none",
    )
    
    print("\nTAPT Configuration:")
    print(f"  Epochs: {tapt_config['num_train_epochs']}")
    print(f"  Batch size: {tapt_config['per_device_train_batch_size']}")
    print(f"  Learning rate: {tapt_config['learning_rate']}")
    print(f"  Masking probability: 15%")
    
    # Trainer setup for MLM
    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_training_args,
        train_dataset=mlm_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\nStarting TAPT (MLM) training...")
    print("This adapts the model to your domain-specific texts.\n")
    
    # Train
    mlm_trainer.train()
    
    # Save
    mlm_trainer.save_model(tapt_output_dir)
    tokenizer.save_pretrained(tapt_output_dir)
    
    print(f"\n✓ TAPT completed and saved to: {tapt_output_dir}")
    print(f"  This model is now adapted to your financial sentiment task texts.")
    
    return tapt_output_dir


# =============================================================================
# SECTION 4: LoRA MODEL SETUP
# =============================================================================

def setup_lora_model(model, lora_config: dict):
    """
    Setup LoRA (Low-Rank Adaptation) configuration and apply to model.
    
    LoRA reduces the number of trainable parameters by adding small
    trainable rank decomposition matrices to existing weights.
    
    Benefits:
    - Much fewer trainable parameters (~0.3% of original)
    - Lower memory requirements
    - Faster training
    - Can be easily added/removed from base model
    
    Args:
        model: Pre-trained model to apply LoRA to
        lora_config: Dictionary with LoRA configuration:
            - r: LoRA rank (lower = fewer parameters)
            - lora_alpha: Scaling factor (usually 2*r)
            - target_modules: Which layers to apply LoRA to
            - lora_dropout: Dropout for LoRA layers
            - bias: How to handle bias terms
    
    Returns:
        Model with LoRA adapters applied
    """
    print("Setting up LoRA configuration...")
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config['r'],                          # Rank of the low-rank matrices
        lora_alpha=lora_config['lora_alpha'],        # Scaling parameter
        target_modules=lora_config['target_modules'],# Which layers to modify
        lora_dropout=lora_config['lora_dropout'],    # Dropout for regularization
        bias=lora_config['bias'],                    # Bias handling strategy
        task_type="SEQ_CLS",                         # Sequence classification task
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameter statistics
    print("\nLoRA Model Statistics:")
    model.print_trainable_parameters()
    
    return model


# =============================================================================
# SECTION 4: MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """
    Main training function - orchestrates the entire pipeline.
    
    Pipeline steps:
    1. Load configuration
    2. Create output directory
    3. Load and split datasets
    4. Load tokenizer and model
    5. Apply LoRA adapters
    6. Tokenize datasets
    7. Setup trainer
    8. Train model
    9. Evaluate on test set
    10. Save model and metrics
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Load Configuration
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING CONFIGURATION")
    print("="*80)
    
    config_path = Path("config.yaml")
    config = load_config(config_path)
    
    # -------------------------------------------------------------------------
    # Step 2: Setup Output Directory
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: SETTING UP OUTPUT DIRECTORY")
    print("="*80)
    
    # Create unique output directory based on model name to avoid overwriting
    base_output_dir = Path(config['training']['output_dir'])
    model_name = config['model']['name'].replace('/', '_').replace('\\', '_')
    output_dir = base_output_dir.parent / f"{base_output_dir.name}-{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Model checkpoints and results will be saved here.")
    
    # -------------------------------------------------------------------------
    # Step 3: Load and Split Datasets
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: LOADING AND SPLITTING DATASETS")
    print("="*80)
    
    datasets = create_datasets(
        data_file=Path(config['data']['train_file']),
        text_column=config['data']['text_column'],
        label_column=config['data']['label_column'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        seed=config['training']['seed']
    )
    
    print(f"\nDataset split summary:")
    print(f"  Train samples: {len(datasets['train'])}")
    print(f"  Validation samples: {len(datasets['validation'])}")
    print(f"  Test samples: {len(datasets['test'])}")
    
    # -------------------------------------------------------------------------
    # Step 4: Load Tokenizer and Model
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: LOADING TOKENIZER AND MODEL")
    print("="*80)
    
    # Check if authentication token is provided (for private models)
    use_auth_token = config.get('model', {}).get('token', None)
    if use_auth_token in [None, 'null', 'None', '']:
        use_auth_token = None
        print("No authentication token provided.")
        print("If this is a private model, set token in config.yaml")
    else:
        print(f"Using authentication token for private model...")
        print(f"Token starts with: {str(use_auth_token)[:10]}...")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        token=use_auth_token
    )
    print("✓ Tokenizer loaded")
    
    # -------------------------------------------------------------------------
    # Step 4.5: OPTIONAL TAPT STAGE
    # -------------------------------------------------------------------------
    tapt_enabled = config.get('tapt_mlm', {}).get('enabled', False)
    model_to_load = config['model']['name']  # Default: use base model
    
    if tapt_enabled:
        print("\n" + "="*80)
        print("STEP 4.5: TAPT (TASK-ADAPTIVE PRE-TRAINING)")
        print("="*80)
        
        print("TAPT is enabled. Running MLM on task texts...")
        
        # Extract all texts from datasets (no labels needed for TAPT)
        # Convert Dataset columns to lists before concatenation
        all_texts = []
        all_texts.extend(list(datasets['train'][config['data']['text_column']]))
        all_texts.extend(list(datasets['validation'][config['data']['text_column']]))
        all_texts.extend(list(datasets['test'][config['data']['text_column']]))
        
        print(f"Using {len(all_texts)} texts for TAPT")
        
        # Run TAPT
        tapt_model_path = run_tapt_mlm(
            model_name=config['model']['name'],
            tokenizer=tokenizer,
            texts=all_texts,
            tapt_config=config['tapt_mlm'],
            use_auth_token=use_auth_token
        )
        
        # Update model path to load TAPT-adapted model
        model_to_load = str(tapt_model_path)
        print(f"\n✓ Will now load TAPT-adapted model from: {model_to_load}")
    else:
        print("\n⊘ TAPT is disabled (tapt_mlm.enabled = false)")
        print("  Skipping TAPT stage, will use base model directly.")
    
    # Load model for sequence classification
    print(f"\nLoading model for sequence classification: {model_to_load}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_to_load,
        num_labels=config['model']['num_labels'],
        token=use_auth_token if model_to_load == config['model']['name'] else None
    )
    
    print(f"✓ Model loaded successfully!")
    print(f"  Model has {config['model']['num_labels']} output classes (labels)")
    
    if tapt_enabled:
        print(f"  Model is TAPT-adapted (continued pre-training on your texts)")
    
    # -------------------------------------------------------------------------
    # Step 5: Apply LoRA Adapters
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: APPLYING LoRA ADAPTERS")
    print("="*80)
    
    model = setup_lora_model(model, config['lora'])
    
    # -------------------------------------------------------------------------
    # Step 6: Tokenize Datasets
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: TOKENIZING DATASETS")
    print("="*80)
    
    print(f"Tokenizing with max_length={config['model']['max_length']}...")
    
    # Remove text column after tokenization, keep labels
    columns_to_remove = [col for col in datasets['train'].column_names 
                        if col != config['data']['label_column']]
    
    tokenized_datasets = datasets.map(
        lambda examples: preprocess_function(
            examples, 
            tokenizer, 
            config['data']['text_column'], 
            config['data']['label_column'], 
            config['model']['max_length']
        ),
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing"
    )
    
    print("Tokenization complete!")
    
    # -------------------------------------------------------------------------
    # Step 7: Setup Data Collator and Training Arguments
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: CONFIGURING TRAINER")
    print("="*80)
    
    # Data collator handles dynamic padding for efficient batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Setup training arguments with type conversion for robustness
    train_cfg = config['training']
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        # Training schedule
        num_train_epochs=float(train_cfg['num_train_epochs']),
        per_device_train_batch_size=int(train_cfg['per_device_train_batch_size']),
        per_device_eval_batch_size=int(train_cfg['per_device_eval_batch_size']),
        gradient_accumulation_steps=int(train_cfg['gradient_accumulation_steps']),
        
        # Optimization
        learning_rate=float(train_cfg['learning_rate']),
        weight_decay=float(train_cfg['weight_decay']),
        warmup_steps=int(train_cfg['warmup_steps']),
        
        # Logging and evaluation
        logging_steps=int(train_cfg['logging_steps']),
        eval_strategy=train_cfg['eval_strategy'],
        save_strategy=train_cfg['save_strategy'],
        save_total_limit=int(train_cfg['save_total_limit']),
        
        # Best model selection
        load_best_model_at_end=bool(train_cfg['load_best_model_at_end']),
        metric_for_best_model=train_cfg['metric_for_best_model'],
        greater_is_better=bool(train_cfg['greater_is_better']),
        
        # Performance optimization
        fp16=bool(train_cfg['fp16']),
        dataloader_pin_memory=bool(train_cfg['dataloader_pin_memory']),
        
        # Reproducibility
        seed=int(train_cfg['seed']),
        
        # Disable external logging
        report_to="none",
    )
    
    print("Training configuration:")
    print(f"  Epochs: {train_cfg['num_train_epochs']}")
    print(f"  Batch size: {train_cfg['per_device_train_batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Warmup steps: {train_cfg['warmup_steps']}")
    print(f"  FP16: {train_cfg['fp16']}")
    
    # -------------------------------------------------------------------------
    # Step 8: Initialize Trainer
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 8: INITIALIZING TRAINER")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Trainer initialized successfully!")
    
    # -------------------------------------------------------------------------
    # Step 9: Train Model
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 9: TRAINING MODEL")
    print("="*80)
    print("\nStarting training... (this may take 45-60 minutes on RTX 4060 8GB)")
    print("Progress will be shown below:\n")
    
    train_result = trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Step 10: Save Model and Tokenizer
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 10: SAVING MODEL AND TOKENIZER")
    print("="*80)
    
    print(f"Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Model and tokenizer saved successfully!")
    
    # -------------------------------------------------------------------------
    # Step 11: Evaluate on Test Set
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 11: EVALUATING ON TEST SET")
    print("="*80)
    
    print("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(tokenized_datasets['test'])
    
    print("\nTest Set Results:")
    print("-" * 40)
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    
    # -------------------------------------------------------------------------
    # Step 12: Save Training Metrics
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 12: SAVING METRICS AND STATE")
    print("="*80)
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    print("Training metrics saved!")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel saved to: {output_dir}")
    print(f"\nYou can now use this model for inference with:")
    print(f"  python src/inference.py --model_path {output_dir}")
    print("\n" + "="*80 + "\n")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    This checks for CUDA availability and starts the training pipeline.
    """
    
    # Print header
    print("\n" + "="*80)
    print("LoRA FINE-TUNING FOR FINANCIAL SENTIMENT ANALYSIS")
    print("="*80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ WARNING: No GPU detected. Training will be slow on CPU.")
        print("Consider using a GPU for faster training.")
    
    # Start training pipeline
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user (Ctrl+C)")
        print("Partial results may have been saved.")
    except Exception as e:
        print(f"\n\n✗ ERROR: Training failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        raise

