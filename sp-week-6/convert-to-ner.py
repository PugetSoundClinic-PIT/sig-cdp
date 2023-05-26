import pandas as pd

FILES = [
    "annotations_alameda.jsonl",
    "annotations_louisville.jsonl",
    "annotations_oakland.jsonl",
    "annotations_seattle.jsonl",
]

for f in FILES:
    sc_df = pd.read_json(f, orient="records", lines=True)

    # Swap to NER
    sc_df["_view_id"] = "ner_manual"

    # Add column I truly don't know what it means
    sc_df["_is_binary"] = False

    # Save to new file
    sc_df.to_json(f.replace(".jsonl", "_ner.jsonl"), orient="records", lines=True)
