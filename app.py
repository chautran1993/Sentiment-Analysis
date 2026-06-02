"""Standalone Streamlit demo for Amazon Electronics sentiment analysis."""

from __future__ import annotations

import os
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import streamlit as st

os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import contractions
except Exception:
    contractions = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.classes.__path__ = []
except Exception:
    torch = None
    nn = None
    F = None

try:
    from transformers import BertForSequenceClassification, BertTokenizerFast
except Exception:
    BertForSequenceClassification = None
    BertTokenizerFast = None

try:
    from keras.models import model_from_json
    from keras.utils import pad_sequences
except Exception:
    model_from_json = None
    pad_sequences = None

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "i", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "with",
    }

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = BASE_DIR / "models"
PARENT_MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else PARENT_MODEL_DIR
ARTIFACT_DIR = BASE_DIR / "artifacts" if (BASE_DIR / "artifacts").exists() else BASE_DIR.parent / "artifacts"
CNN_PATH = MODEL_DIR / "cnn_model.pt"
LSTM_PATH = MODEL_DIR / "lstm_model.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
BERT_DIR = MODEL_DIR / "bert"
KERAS_CNN_CONFIG = MODEL_DIR / "cnn" / "model_config.json"
KERAS_CNN_WEIGHTS = MODEL_DIR / "cnn" / "best_model.weights.h5"
KERAS_LSTM_CONFIG = MODEL_DIR / "lstm" / "model_config.json"
KERAS_LSTM_WEIGHTS = MODEL_DIR / "lstm" / "best_model.weights.h5"
KERAS_CNN_TOKENIZER_PATH = ARTIFACT_DIR / "cnn" / "keras_tokenizer.pkl"
KERAS_LSTM_TOKENIZER_PATH = ARTIFACT_DIR / "keras_tokenizer.pkl"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_VI = {"Negative": "Tiêu cực", "Neutral": "Trung lập", "Positive": "Tích cực"}
COLORS = {"Negative": "#e74c3c", "Neutral": "#95a5a6", "Positive": "#2ecc71"}
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

MAX_LEN_CLASSIC = 150
MAX_LEN_BERT = 128
NEGATIVE_WORDS = {"broke", "broken", "terrible", "worst", "waste", "disappointed", "awful", "bad", "poor", "refund", "defective", "failed", "dead"}
POSITIVE_WORDS = {"great", "excellent", "love", "loved", "perfect", "amazing", "good", "best", "awesome", "fantastic", "works", "recommend", "happy"}


if nn is not None:

    class SentimentCNN(nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
            self.conv1 = nn.Conv1d(128, 128, kernel_size=5)
            self.conv2 = nn.Conv1d(128, 64, kernel_size=5)
            self.fc1 = nn.Linear(64, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x).permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)


    class SentimentBiLSTM(nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
            self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=0.2)
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)
            _, (hidden, _) = self.lstm(x)
            x = torch.cat((hidden[-2], hidden[-1]), dim=1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)



@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: list[float]

    @property
    def confidence(self) -> float:
        return max(self.probabilities)


@dataclass
class LoadedModel:
    name: str
    key: str
    is_real: bool
    predict: Callable[[str], Prediction]


st.set_page_config(
    page_title="Phân tích Cảm xúc Bình luận Amazon Electronics",
    page_icon="🛒",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px; }
        .stApp { background: #f7f9fb; color: #1f2937; }
        section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.25rem;
        }
        .model-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            min-height: 132px;
            margin-bottom: 0.6rem;
        }
        .model-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.45rem; }
        .pred-label {
            display: inline-block;
            color: #ffffff;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .confidence { font-size: 1.55rem; font-weight: 750; margin-top: 0.45rem; }
        .badge-real, .badge-demo {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .badge-real { color: #14532d; background: #dcfce7; }
        .badge-demo { color: #7c2d12; background: #ffedd5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize(text: str) -> str:
    text = text.lower()
    if contractions is not None:
        text = contractions.fix(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_keras_text(text: str) -> str:
    """Mirror the CNN/BiLSTM training preprocessing as closely as possible."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = re.sub(r"\s+", " ", text).strip().split()
    tokens = [
        token
        for token in tokens
        if token.isalpha() and token not in ENGLISH_STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def tokenize_classic(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [int(vocab.get(token, 1)) for token in normalize(text).split()[:MAX_LEN_CLASSIC]]
    return ids + [0] * (MAX_LEN_CLASSIC - len(ids))


def mock_predictor(name: str) -> Callable[[str], Prediction]:
    def predict(text: str) -> Prediction:
        tokens = set(normalize(text).split())
        neg_hits = len(tokens & NEGATIVE_WORDS)
        pos_hits = len(tokens & POSITIVE_WORDS)
        if neg_hits > pos_hits:
            probs = np.array([0.76, 0.16, 0.08])
        elif pos_hits > neg_hits:
            probs = np.array([0.08, 0.16, 0.76])
        elif neg_hits and pos_hits:
            probs = np.array([0.34, 0.42, 0.24]) if name != "BERT" else np.array([0.31, 0.36, 0.33])
        else:
            probs = np.array([0.18, 0.64, 0.18])
        probs = probs / probs.sum()
        return Prediction(LABELS[int(probs.argmax())], probs.round(4).tolist())

    return predict


def load_vocab() -> dict[str, int]:
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("vocab.json must be a word-to-index dictionary.")
    return {str(k): int(v) for k, v in data.items()}


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location=DEVICE)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} is not a state_dict.")
    return {key.removeprefix("module."): value for key, value in loaded.items()}


def load_keras_tokenizer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def keras_embedding_input_dim(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for layer in config["config"]["layers"]:
        if layer.get("class_name") == "Embedding":
            return int(layer["config"]["input_dim"])
    raise ValueError("Embedding layer not found in Keras config.")


def classic_predictor(model: nn.Module, vocab: dict[str, int]) -> Callable[[str], Prediction]:
    model.eval()

    def predict(text: str) -> Prediction:
        x = torch.tensor([tokenize_classic(text, vocab)], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).detach().cpu().numpy()[0]
        return Prediction(LABELS[int(probs.argmax())], probs.round(4).tolist())

    return predict


def keras_predictor(model, tokenizer) -> Callable[[str], Prediction]:
    def predict(text: str) -> Prediction:
        clean_text = preprocess_keras_text(text)
        sequence = tokenizer.texts_to_sequences([clean_text])
        x = pad_sequences(sequence, maxlen=256, padding="post", truncating="post")
        probs_obj = model(x)
        if hasattr(probs_obj, "detach"):
            probs = probs_obj.detach().cpu().numpy()[0]
        else:
            probs = np.asarray(probs_obj)[0]
        return Prediction(LABELS[int(np.argmax(probs))], probs.round(4).tolist())

    return predict


def load_keras_model(config_path: Path, weights_path: Path):
    if model_from_json is None or pad_sequences is None:
        raise RuntimeError("Keras is not available.")
    model = model_from_json(config_path.read_text(encoding="utf-8"))
    model.load_weights(str(weights_path))
    return model


def fallback(name: str, key: str) -> LoadedModel:
    return LoadedModel(name=name, key=key, is_real=False, predict=mock_predictor(name))


def load_cnn(vocab: dict[str, int]) -> LoadedModel:
    try:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is not available.")
        model = SentimentCNN(max(vocab.values(), default=1) + 1).to(DEVICE)
        model.load_state_dict(load_state_dict(CNN_PATH), strict=True)
        return LoadedModel("CNN", "cnn", True, classic_predictor(model, vocab))
    except Exception:
        try:
            model = load_keras_model(KERAS_CNN_CONFIG, KERAS_CNN_WEIGHTS)
            tokenizer = load_keras_tokenizer(KERAS_CNN_TOKENIZER_PATH)
            if len(tokenizer.word_index) + 1 != keras_embedding_input_dim(KERAS_CNN_CONFIG):
                raise RuntimeError("CNN tokenizer does not match the saved CNN embedding vocabulary.")
            return LoadedModel("CNN", "cnn", True, keras_predictor(model, tokenizer))
        except Exception:
            return fallback("CNN", "cnn")


def load_lstm(vocab: dict[str, int]) -> LoadedModel:
    try:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is not available.")
        model = SentimentBiLSTM(max(vocab.values(), default=1) + 1).to(DEVICE)
        model.load_state_dict(load_state_dict(LSTM_PATH), strict=True)
        return LoadedModel("BiLSTM", "lstm", True, classic_predictor(model, vocab))
    except Exception:
        try:
            model = load_keras_model(KERAS_LSTM_CONFIG, KERAS_LSTM_WEIGHTS)
            tokenizer = load_keras_tokenizer(KERAS_LSTM_TOKENIZER_PATH)
            return LoadedModel("BiLSTM", "lstm", True, keras_predictor(model, tokenizer))
        except Exception:
            return fallback("BiLSTM", "lstm")


def load_bert() -> LoadedModel:
    try:
        if BertForSequenceClassification is None or BertTokenizerFast is None or torch is None:
            raise RuntimeError("transformers/PyTorch is not available.")
        tokenizer_source = BERT_DIR if (BERT_DIR / "tokenizer_config.json").exists() else "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_source, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(BERT_DIR, num_labels=3, local_files_only=True).to(DEVICE)
        model.eval()

        def predict(text: str) -> Prediction:
            encoded = tokenizer(
                text.lower(),
                max_length=MAX_LEN_BERT,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            with torch.no_grad():
                probs = torch.softmax(model(**encoded).logits, dim=1).detach().cpu().numpy()[0]
            return Prediction(LABELS[int(probs.argmax())], probs.round(4).tolist())

        return LoadedModel("BERT", "bert", True, predict)
    except Exception:
        return fallback("BERT", "bert")


def model_file_signature() -> tuple[tuple[str, int, int] | tuple[str, None, None], ...]:
    """Invalidate Streamlit's model cache when local model files change."""
    paths = [
        CNN_PATH,
        LSTM_PATH,
        VOCAB_PATH,
        KERAS_CNN_CONFIG,
        KERAS_CNN_WEIGHTS,
        KERAS_CNN_TOKENIZER_PATH,
        KERAS_LSTM_CONFIG,
        KERAS_LSTM_WEIGHTS,
        KERAS_LSTM_TOKENIZER_PATH,
        BERT_DIR / "config.json",
        BERT_DIR / "model.safetensors",
        BERT_DIR / "pytorch_model.bin",
        BERT_DIR / "tokenizer.json",
        BERT_DIR / "tokenizer_config.json",
    ]
    signature = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_models(_signature: tuple[tuple[str, int, int] | tuple[str, None, None], ...]) -> list[LoadedModel]:
    try:
        vocab = load_vocab()
        cnn = load_cnn(vocab)
        lstm = load_lstm(vocab)
    except Exception:
        try:
            cnn = LoadedModel(
                "CNN",
                "cnn",
                True,
                keras_predictor(load_keras_model(KERAS_CNN_CONFIG, KERAS_CNN_WEIGHTS), load_keras_tokenizer(KERAS_CNN_TOKENIZER_PATH)),
            )
            tokenizer = load_keras_tokenizer(KERAS_CNN_TOKENIZER_PATH)
            if len(tokenizer.word_index) + 1 != keras_embedding_input_dim(KERAS_CNN_CONFIG):
                raise RuntimeError("CNN tokenizer does not match the saved CNN embedding vocabulary.")
        except Exception:
            cnn = fallback("CNN", "cnn")
        try:
            lstm = LoadedModel(
                "BiLSTM",
                "lstm",
                True,
                keras_predictor(load_keras_model(KERAS_LSTM_CONFIG, KERAS_LSTM_WEIGHTS), load_keras_tokenizer(KERAS_LSTM_TOKENIZER_PATH)),
            )
        except Exception:
            lstm = fallback("BiLSTM", "lstm")
    return [cnn, lstm, load_bert()]


def prob_chart(pred: Prediction, height: int = 150) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=pred.probabilities,
            y=LABELS,
            orientation="h",
            marker_color=[COLORS[label] for label in LABELS],
            text=[f"{p:.1%}" for p in pred.probabilities],
            textposition="auto",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=8, b=4),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(categoryorder="array", categoryarray=LABELS[::-1]),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def comparison_chart(predictions: dict[str, Prediction]) -> go.Figure:
    fig = go.Figure()
    for label in LABELS:
        fig.add_bar(
            name=LABEL_VI[label],
            x=list(predictions.keys()),
            y=[predictions[name].probabilities[LABELS.index(label)] for name in predictions],
            marker_color=COLORS[label],
            hovertemplate=f"{LABEL_VI[label]}: %{{y:.1%}}<extra></extra>",
        )
    fig.update_layout(
        barmode="group",
        height=360,
        yaxis=dict(range=[0, 1], tickformat=".0%", title="Xác suất"),
        xaxis=dict(title="Mô hình"),
        legend_title_text="Lớp",
        margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def macro_f1_chart() -> go.Figure:
    chart_rows = {
        "Baseline": [0.6349, 0.6856, 0.7601],
        "SMOTE": [0.6129, 0.6696, 0.7566],
        "Class Weights": [0.6503, 0.6841, 0.7585],
    }
    colors = {"Baseline": "#3498db", "SMOTE": "#9b59b6", "Class Weights": "#2ecc71"}
    fig = go.Figure()
    for strategy, values in chart_rows.items():
        fig.add_bar(
            name=strategy,
            x=["CNN", "BiLSTM", "BERT"],
            y=values,
            marker_color=colors[strategy],
            text=[f"{value:.1%}" for value in values],
            textposition="outside",
            hovertemplate=f"{strategy}<br>%{{x}}: %{{y:.1%}}<extra></extra>",
        )
    fig.update_layout(
        barmode="group",
        height=330,
        yaxis=dict(range=[0, 0.9], tickformat=".0%", title="Macro F1"),
        xaxis=dict(title="Mô hình"),
        margin=dict(l=20, r=20, t=35, b=30),
        legend_title_text="Phương pháp",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def distribution_chart() -> go.Figure:
    counts = [19075, 7174, 73751]
    total = sum(counts)
    fig = go.Figure(
        go.Bar(
            x=["Tiêu cực", "Trung lập", "Tích cực"],
            y=counts,
            marker_color=[COLORS[label] for label in LABELS],
            text=[f"{count / total:.1%}" for count in counts],
            textposition="outside",
            hovertemplate="%{x}: %{y:,} review (%{text})<extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        yaxis=dict(title="Số review", range=[0, 80000]),
        margin=dict(l=20, r=20, t=35, b=30),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def render_sidebar(models: list[LoadedModel]) -> None:
    st.sidebar.title("NLP Sentiment Demo")
    st.sidebar.caption("Amazon Electronics Reviews")
    st.sidebar.markdown("**Môn học:** Lập trình Python cho Máy học")
    st.sidebar.markdown("**Lớp:** CS116.F21.CN2.TTNT")
    st.sidebar.markdown("**Nhóm:** 3")
    st.sidebar.markdown("**GVHD:** ThS Nguyễn Hữu Quyền")
    st.sidebar.divider()
    st.sidebar.markdown("**Trạng thái mô hình**")
    for model in models:
        badge_class = "badge-real" if model.is_real else "badge-demo"
        badge_text = "REAL ✓" if model.is_real else "DEMO ⚠️"
        st.sidebar.markdown(f"{model.name}: <span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.caption(f"Thiết bị inference: `{DEVICE}`")


def set_example(text: str) -> None:
    st.session_state.review_text = text


def render_predict_tab(models: list[LoadedModel]) -> None:
    examples = {
        "Tích cực": "Excellent headphones with great sound, long battery life, and easy setup.",
        "Tiêu cực": "The device arrived defective and never worked properly.",
        "Trung lập": "The keyboard is okay for basic typing, with average build quality and normal packaging.",
    }
    if "review_text" not in st.session_state:
        st.session_state.review_text = examples["Tích cực"]

    st.subheader("Nhập bình luận cần phân tích")
    cols = st.columns(len(examples))
    for col, (label, text) in zip(cols, examples.items()):
        with col:
            st.button(label, use_container_width=True, on_click=set_example, args=(text,))

    text = st.text_area(
        "Review",
        key="review_text",
        label_visibility="collapsed",
        height=150,
        placeholder="Dán bình luận Amazon Electronics vào đây...",
    )
    active_models = [model for model in models if model.is_real]
    if not st.button("Phân tích", type="primary", use_container_width=True):
        st.info("Nhấn **Phân tích** để chạy CNN, BiLSTM và BERT trên cùng một review.")
        return
    if not text.strip():
        st.warning("Vui lòng nhập một bình luận trước khi phân tích.")
        return

    predictions = {model.name: model.predict(text) for model in active_models}
    st.subheader("Kết quả từng mô hình")
    cols = st.columns(3)
    for col, model in zip(cols, models):
        with col:
            if model.name not in predictions:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-title">{model.name}</div>
                        <span class="pred-label" style="background:#95a5a6;">Chưa có model thật</span>
                        <div class="confidence">--</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info("Thiếu checkpoint/tokenizer khớp, nên không dùng mock để kết luận.")
            else:
                pred = predictions[model.name]
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-title">{model.name}</div>
                        <span class="pred-label" style="background:{COLORS[pred.label]};">{LABEL_VI[pred.label]}</span>
                        <div class="confidence">{pred.confidence:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(prob_chart(pred), use_container_width=True, key=f"{model.key}_prob")

    if len(predictions) < len(models):
        st.caption("Một số mô hình không có đủ artifact thật/khớp nên không được dùng trong dự đoán live.")
    elif len({pred.label for pred in predictions.values()}) > 1:
        st.caption("⚠️ Các mô hình không đồng thuận - thường gặp ở bình luận có sắc thái không rõ ràng.")
    else:
        st.caption("Các mô hình đang đồng thuận trên review này.")

    if predictions:
        st.subheader("So sánh xác suất giữa các mô hình")
        st.plotly_chart(comparison_chart(predictions), use_container_width=True)


def render_results_tab() -> None:
    st.subheader("Kết quả trên tập test")
    results = [
        {"Phương pháp": "Baseline", "Model": "CNN", "Accuracy": "87.5%", "Macro F1": "63.5%", "Precision": "74.4%", "Recall": "64.0%"},
        {"Phương pháp": "Baseline", "Model": "BiLSTM", "Accuracy": "88.3%", "Macro F1": "68.6%", "Precision": "72.2%", "Recall": "67.5%"},
        {"Phương pháp": "Baseline", "Model": "BERT", "Accuracy": "90.6%", "Macro F1": "76.0%", "Precision": "76.1%", "Recall": "75.9%"},
        {"Phương pháp": "SMOTE", "Model": "CNN", "Accuracy": "86.2%", "Macro F1": "61.3%", "Precision": "69.5%", "Recall": "62.8%"},
        {"Phương pháp": "SMOTE", "Model": "BiLSTM", "Accuracy": "86.8%", "Macro F1": "67.0%", "Precision": "68.5%", "Recall": "66.1%"},
        {"Phương pháp": "SMOTE", "Model": "BERT", "Accuracy": "90.5%", "Macro F1": "75.7%", "Precision": "75.8%", "Recall": "75.5%"},
        {"Phương pháp": "Class Weights", "Model": "CNN", "Accuracy": "78.2%", "Macro F1": "65.0%", "Precision": "67.9%", "Recall": "72.1%"},
        {"Phương pháp": "Class Weights", "Model": "BiLSTM", "Accuracy": "81.8%", "Macro F1": "68.4%", "Precision": "67.6%", "Recall": "74.6%"},
        {"Phương pháp": "Class Weights", "Model": "BERT", "Accuracy": "89.9%", "Macro F1": "75.9%", "Precision": "74.8%", "Recall": "77.3%"},
    ]
    st.dataframe(results, use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Macro F1 theo mô hình và phương pháp xử lý mất cân bằng**")
        st.plotly_chart(macro_f1_chart(), use_container_width=True)
    with col2:
        st.markdown("**Phân bố nhãn trong mẫu 100.000 review**")
        st.plotly_chart(distribution_chart(), use_container_width=True)
    st.markdown(
        """
        **Ghi chú dữ liệu:** dataset = Amazon Electronics Reviews, lấy mẫu ngẫu nhiên 100.000 review bằng reservoir sampling.
        Phân phối nhãn giữ theo dữ liệu gốc: Negative 19.1%, Neutral 7.2%, Positive 73.8%.
        Train/Val/Test = 70k/15k/15k, stratified; đánh giá trên test set bằng macro F1.

        **Kết luận chính:** BERT đứng đầu ở cả 3 cấu hình. Baseline cho BERT tốt nhất về Macro F1 (76.0%),
        trong khi Class Weights tăng Recall macro nhưng làm giảm Accuracy do model chú ý nhiều hơn tới lớp hiếm.
        Neutral là lớp khó nhất vì chỉ chiếm khoảng 7.2% dữ liệu.
        """
    )


def main() -> None:
    inject_css()
    models = load_models(model_file_signature())
    render_sidebar(models)
    st.title("Phân tích Cảm xúc Bình luận Amazon Electronics")
    st.caption("So sánh CNN, BiLSTM và BERT cho bài toán 3 lớp: Negative / Neutral / Positive.")
    tab_predict, tab_results = st.tabs(["Dự đoán & So sánh", "Kết quả & Dữ liệu"])
    with tab_predict:
        render_predict_tab(models)
    with tab_results:
        render_results_tab()


if __name__ == "__main__":
    main()
