# Zero-shot demo

The repository intentionally does not redistribute clinical images. Use a
locally available, de-identified image that you are authorized to process.

First accept the MedSigLIP model terms on Hugging Face and authenticate with
`hf auth login`. Then run:

```bash
python scripts/zero_shot.py \
  --image /path/to/deidentified_cxr.png \
  --labels cardiomegaly "pleural effusion" pneumothorax "pulmonary edema" \
  --output examples/zero_shot_output.json
```

The script creates a positive and negative prompt for every finding and
normalizes each prompt pair independently. This avoids treating potentially
co-occurring chest findings as a mutually exclusive multiclass problem.

See `zero_shot_output.example.json` for the output schema. Its probabilities
are illustrative placeholders, not model results or clinical claims.
