#!/usr/bin/env python

import pandas as pd
import spacy

# Load IRR dataset
df = pd.read_json(
    "local-interest-groups-irr-annotation-set.jsonl",
    orient="records",
    lines=True,
)

# Load the model
nlp = spacy.load("./model/model-best")

print("Evaluating IRR data for entities...")
print("=" * 80)

# Iter each example and check for ents
processed = []
for _, row in df.iterrows():
    # Process
    doc = nlp(row.text)

    # Append results to list
    print("Original text:")
    print(f"\t{row.text}")
    print()

    # Check for spans
    for ent in doc.ents:
        print(ent.label_)
        print(ent.text)
    
    print("-" * 80)
    print()
    print()
