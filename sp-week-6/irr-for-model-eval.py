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

# Iter each example and check for spans
processed = []
for _, row in df.iterrows():
    # Process
    doc = nlp(row.text)

    # Check for spans
    if len(doc.spans["sc"]) > 0:
        spans = doc.spans["sc"]
    else:
        spans = None
    
    # Append results to list
    processed.append({
        "original-text": row.text,
        "spans": spans,
    })

# Convert to DataFrame
spans_data = pd.DataFrame(processed)

# Calculate eval features
n_docs_with_spans = len(spans_data.loc[~spans_data.spans.isnull()])

# Print results
print("Extra eval results:")
print("-" * 40)
print(f"n docs with spans: {n_docs_with_spans}")
