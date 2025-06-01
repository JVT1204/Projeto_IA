# app.py

import json
import streamlit as st
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import preprocess_text

st.set_page_config(
    page_title="Classificador de Notícias de Futebol",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📰 Classificador de Notícias de Futebol")
st.markdown(
    """
    Este aplicativo utiliza um modelo BERT em português para categorizar notícias de futebol
    em: **resultado**, **transferência**, **lesão**, **tática** ou **outras**.
    """
)

@st.cache_resource(show_spinner=False)
def load_tokenizer_model(model_dir: str):
    """
    Carrega e retorna o tokenizer e o modelo a partir de um diretório Hugging Face salvo.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_category_map(json_path: str):
    """
    Carrega o mapeamento id->categoria de um arquivo JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # mapping['id_to_category'] deve ser um dicionário { "0": "resultado", "1": "transferencia", ... }
    return mapping["id_to_category"]

# --- Ajuste aqui: apontar para a mesma pasta usada no notebook ---
MODEL_DIR = "models/best_model"
CATEGORIES_JSON = "data/processed/category_mapping.json"

tokenizer, model, device = load_tokenizer_model(MODEL_DIR)
id_to_category = load_category_map(CATEGORIES_JSON)

def predict_category(text: str):
    """
    Recebe texto cru, faz preprocess, tokeniza, roda o modelo e retorna:
        - categoria_prevista (string)
        - probabilidades (np.array de tamanho num_labels)
    """
    # ETAPA 1: pré-processar
    processed = preprocess_text(text)

    # Se, após limpar, o texto ficar vazio, não classifica
    if processed.strip() == "":
        return None, None

    # ETAPA 2: tokenização
    encoding = tokenizer(
        processed,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # ETAPA 3: inferência
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits.squeeze()  # shape: (num_labels,)
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    # ETAPA 4: obter índice e categoria
    idx = int(np.argmax(probs))
    category = id_to_category[str(idx)]
    return category, probs

# ---------------------------
# Interface do usuário
# ---------------------------

st.subheader("Insira o título ou trecho da notícia abaixo:")
user_input = st.text_area("", placeholder="Cole aqui o título ou resumo da notícia...", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("Classificar"):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Por favor, insira um texto antes de classificar.")
        else:
            with st.spinner("Classificando…"):
                categoria, probabilidades = predict_category(user_input)

            if categoria is None:
                st.error("Não foi possível processar o texto. Tente inserir algo diferente.")
            else:
                st.success(f"**Categoria prevista:** {categoria.upper()}")
                # Mostrar também as probabilidades para cada classe, ordenadas
                st.markdown("**Probabilidades por categoria:**")
                prob_list = []
                for key, cat in id_to_category.items():
                    prob_list.append((cat, float(probabilidades[int(key)])))
                prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
                for cat, p in prob_list:
                    st.write(f"- {cat:12s}: {p*100:5.2f}%")

with col2:
    st.image(
        "https://raw.githubusercontent.com/streamlit/logo/master/streamlit-logo-primary-colormark-darktext.png",
        width=200,
        caption="Streamlit"
    )

st.markdown("---")
st.markdown(
    """
    **Observações**  
    - O texto é pré-processado (remoção de stopwords, lematização) antes de entrar no BERT.  
    - Se a mensagem for muito curta ou não contiver termos em português relevantes, ele pode não conseguir classificar.  
    - As categorias possíveis são: `resultado`, `transferência`, `lesão`, `tática` e `outras`.  
    """
)
