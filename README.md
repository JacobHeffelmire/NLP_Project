# NLP Project: Speaker-Independent Factuality Classification

This project investigates whether linguistic features can predict
statement factuality when evaluated on previously unseen speakers.

## Dataset
LIAR2, obtained from Hugging Face (chengxuphd/liar2).

The dataset is filtered into binary true/false classification,
retaining 15,650 statements from 3,558 speakers.

## Models
- Logistic Regression
- Linear Support Vector Machine

## Features
- TF-IDF
- Stylometric features
- Combined TF-IDF and stylometric features

## Evaluation
Models are compared using random and speaker-independent splits.

Metrics include accuracy, macro-precision, macro-recall, and macro-F1.

## Running
Install dependencies:
pip install -r requirements.txt

Run:
python main.py

## Results
Experimental results are stored in results/metrics.csv.
