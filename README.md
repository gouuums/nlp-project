# nlp-project
## ENSAI 3rd Year — Webmining & NLP Course: Final Project Report

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8-orange.svg)](https://spacy.io/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-5.2-green.svg)](https://sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Authors:** Thomas Goumont (`thomas.goumont@eleve.ensai.fr`) & Léo Gruchociak (`leo.gruchociak@eleve.ensai.fr`)  
> **Institution:** ENSAI (École Nationale de la Statistique et de l'Analyse de l'Information)  
> **Course:** Webmining and Natural Language Processing (Pr. Cyrielle Mallart)  

---

## 📋 Overview

This repository contains the final report and code for the 3rd and 4th practical sessions of the **Webmining and NLP** course at ENSAI. 

The main objective of this project is to explore how different text representations capture movie characteristics, semantic relationships, and genre structures. Using a corpus of 499 movie articles extracted from the English Wikipedia API for the release year **2023**, we compare traditional feature engineering (Named Entity Recognition, Jaccard distance) with modern deep learning representations (**Sentence-BERT embeddings**).

The complete analysis, code, and visualizations are embedded within the Jupyter Notebook: `nlp_proj.ipynb`.

---

## 🗂️ Notebook Structure (`nlp_proj.ipynb`)

The notebook is structured into six core analytical parts, followed by a comprehensive appendix:

1. **Data Gathering & Curation via Wikipedia API**
   - Extraction of 500 movie article titles from the `"2023 films"` category.
   - Comparison of MediaWiki API extraction modules (`revisions` vs. `extracts`).
   - Proxy-labeling strategy: mapping user-contributed Wikipedia categories to standardized multi-label movie genres using keyword spotting.
   - Data persistence (`data_films_2023.json`)[cite: 1].

2. **Exploratory Data Analysis (EDA)**
   - Corpus-wide descriptive statistics (article length variance, vocabulary richness)[cite: 1].
   - Linguistic profiling via **spaCy** (`en_core_web_md`): Type-Token Ratio (TTR), sentence length, and Part-of-Speech (POS) distribution analysis (highlighting the encyclopedic nature of the corpus)[cite: 1].
   - Critical evaluation of dataset strengths, biases (popularity/blockbuster bias), and future enrichment strategies (e.g., integrating TMDb or sentiment analysis via Hugging Face Transformers)[cite: 1].

3. **Named Entity Recognition (NER) & Information Extraction**
   - Pipeline optimization by disabling unnecessary components for faster batch inference (`nlp.pipe`)[cite: 1].
   - Extraction and frequency analysis of standard IOB entities (`PERSON`, `ORG`, `GPE`, `DATE`, etc.)[cite: 1].
   - Document-level co-occurrence analysis and identification of editorial patterns vs. semantic links[cite: 1].

4. **Text Embeddings & Semantic Vectorization**
   - Introduction to encoder-only architectures (BERT) and Sub-word tokenization (WordPiece)[cite: 1].
   - Generating 384-dimensional dense semantic vectors using **Sentence-BERT** (`all-MiniLM-L6-v2`) with mean pooling[cite: 1].
   - Comparison against static average word-vector representations[cite: 1].

5. **Entity-Based Distance Matrix & Jaccard Index**
   - Constructing a Jaccard distance matrix based on filtered named entities (removing noise like `CARDINAL`, `DATE`, `ORDINAL`)[cite: 1].
   - Analyzing the "exact match problem" and sparsity issues in entity sets[cite: 1].

6. **Unsupervised Clustering & Genre Recovery**
   - Data-driven hyperparameter tuning for the number of clusters ($k=8$)[cite: 1].
   - **Spectral Clustering** applied to both SBERT semantic features and precomputed NER affinity matrices[cite: 1].
   - Dimensionality reduction and visualization via **t-SNE**[cite: 1].
   - Quantitative evaluation using **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** against Wikipedia proxy-genres[cite: 1].
   - Detailed qualitative comparison through cluster-genre distribution heatmaps[cite: 1].

---

## 📦 Requirements & Installation

The project runs in Python 3.12+ with the following key dependencies[cite: 1]:

```bash
pip install sentence-transformers spacy seaborn tqdm scikit-learn pandas matplotlib
python -m spacy download en_core_web_md
