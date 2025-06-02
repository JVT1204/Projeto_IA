# Classificador Inteligente de Notícias de Futebol

Este projeto implementa um classificador de notícias de futebol utilizando técnicas de Processamento de Linguagem Natural (NLP) e Deep Learning. O sistema categoriza automaticamente notícias em cinco categorias: resultado, transferência, lesão, tática e outras.

## Autores

- Daniel Reis Raske - 10223349
- Eduardo Marui de Camargo - 10400734
- Victor Vergara Marques de Oliveira - 10403378
- Vitor dos Santos Souza - 10204809
- João Vitor Tortorello - 10402674

## Índice de Entregas

### 1. Notebook de Desenvolvimento
- *Arquivo*: Projeto_N2_NoticiasEsporte.ipynb
- *Descrição*: Notebook Python contendo todo o desenvolvimento do modelo, incluindo:
  - Coleta e pré-processamento dos dados
  - Treinamento do modelo BERT
  - Avaliação e métricas
  - Visualizações e análises
- *Como Executar*: 
  bash
  pip install -r requirements.txt
  jupyter notebook Projeto_N2_NoticiasEsporte.ipynb
  

### 2. Aplicação Web (Streamlit)
- *Arquivo*: app.py
- *Descrição*: Interface web interativa para classificação de notícias em tempo real
- *Funcionalidades*:
  - Input de texto para classificação
  - Exibição da categoria prevista
  - Visualização das probabilidades por categoria
- *Como Executar*:
  bash
  pip install -r requirements.txt
  streamlit run app.py
  

### 3. Artigo do Projeto
- *Arquivo*: Projeto_IA/Artigo_NoticiasEsportivasIA.pdf
- *Descrição*: Documento técnico detalhando:
  - Introdução
  - Fundamentação
  - Implementação
  - Conclusão e discussão

### 4. Vídeo de Apresentação
- *Link*: https://youtu.be/yaS5sMjIv28
- *Conteúdo*:
  - Demonstração do projeto
  - Principais funcionalidades
  - Resultados obtidos

## Tecnologias Utilizadas

- Python 3.x
- PyTorch
- Transformers (BERT)
- Streamlit
- Jupyter Notebook
- Scikit-learn
- Pandas
- Numpy

## Como Começar

1. Clone o repositório:
   bash
   git clone https://github.com/JVT1204/Projeto_IA.git
   

2. Instale as dependências:
   bash
   pip install -r requirements.txt
   

3. Execute a aplicação Streamlit:
   bash
   streamlit run app.py
