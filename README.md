# Sentiment Analysis Python

Streamlit demo for Amazon Electronics review sentiment analysis with three models: CNN, BiLSTM, and BERT.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py --server.port 8511
```

## Included

- `app.py`: Streamlit demo app.
- `models/cnn/`: CNN architecture and weights.
- `models/lstm/`: BiLSTM architecture and weights.
- `artifacts/`: tokenizers and label encoders used by CNN/BiLSTM.
- `models/bert/`: BERT config and tokenizer files.

## BERT Weight Note

The BERT weight file `model.safetensors` is about 438 MB, so it is not committed to GitHub. To run BERT in REAL mode, place it here:

```text
models/bert/model.safetensors
```

If the BERT weight is absent, the app still runs safely and falls back to DEMO mode for BERT only.
