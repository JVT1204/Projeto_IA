# src/preprocess.py

import re
import unicodedata
import spacy

# Carrega o pt_core once, na importação do módulo
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    # se não encontrar o modelo 'lg', tenta o 'sm' — mas ideal é ter o lg instalado
    nlp = spacy.load("pt_core_news_sm")

def preprocess_text(text: str) -> str:
    """
    Pré-processa o texto para análise:
      - Lowercase, normaliza acentos
      - Remove URLs, caracteres especiais
      - Tokeniza, remove stopwords, lematiza
    """
    if not isinstance(text, str):
        return ""

    # Normaliza o texto (remove acentos)
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove caracteres especiais (tudo que não seja letra ou número ou underscore vira espaço)
    text = re.sub(r"[^\w\s]", " ", text)

    # Colapsa múltiplos espaços em apenas um
    text = re.sub(r"\s+", " ", text).strip()

    # Agora processa com spaCy já carregado globalmente
    doc = nlp(text)

    # Remove stopwords e pontuação e pega o lemma
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return " ".join(tokens)
