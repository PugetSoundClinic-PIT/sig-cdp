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

print("Evaluating IRR data for spans...")
print("=" * 80)

# Iter each example and check for spans
processed = []
for _, row in df.iterrows():
    # Process
    doc = nlp(row.text)

    # Check for spans
    for span_group in doc.spans:
        longest_span_for_pawo = ""
        longest_span_for_p = ""
        group = doc.spans[span_group]
        for span in group:
            if (
                span.label_ == "PERSON" 
                and len(span.text) > len(longest_span_for_p)
            ):
                longest_span_for_p = span.text
            elif (
                span.label_ == "PERSON-AFFLIATED-WITH-ORG" 
                and len(span.text) > len(longest_span_for_pawo)
            ):
                longest_span_for_pawo = span.text
           
        # Append results to list
        print("Original text:")
        print(f"\t{row.text}")
        print()
        print("PERSON-AFFLIATED-WITH-ORG:")
        print(f"\t{longest_span_for_pawo}")
        print()
        print("PERSON:")
        print(f"\t{longest_span_for_p}")
        print()
        print("-" * 40)
