# Social Spam Detection (DistilBERT)

A text classification project for detecting **spam vs ham** content using a fine-tuned DistilBERT model. The training workflow is provided as a Jupyter notebook and uses the Hugging Face Transformers ecosystem.

> Status: Research/Training notebook included; deployment-ready inference structure recommended (see “Deployment” below).

## Project Overview
This repository contains:
- A Jupyter notebook that loads/cleans labeled text data and trains a spam/ham classifier.  
- Transformer tokenizer artifacts and config files saved from the training process.

The model is intended to classify short text content (e.g., social posts) into:
- `ham` (label 0)
- `spam` (label 1)

## Repository Contents
- `social-spam-clean.ipynb` — end-to-end notebook: data loading, preprocessing, training/evaluation, and saving artifacts.  
- `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json` — tokenizer assets.  
- `config.json` — model configuration.  
- `trainer_state.json` — training metadata/state (optional, mainly for resuming/analysis).

## Quick Start (Inference)
### 1) Setup environment
Recommended: Python 3.10+.

Install dependencies:
```bash
pip install -U transformers torch accelerate datasets scikit-learn pandas
