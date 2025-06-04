# 📚 Analyse et Visualisation de Données : Correction post-OCR avec les LLMs

Ce projet explore, dans le cadre de mon mémoire de master, l'utilisation des **grands modèles de langage (LLMs)** pour corriger automatiquement les erreurs générées par l'OCR

## 🧠 Objectif

Évaluer l’efficacité de différents modèles (GPT, LLaMA, Gemma, DeepSeek...) pour de la correction post-OCR, en comparant leurs performances selon divers prompts, corpus et paramètres (ex. température).

## 🔍 Méthodologie

- Reproduction et adaptation de l’approche de **Boros et al.** (2024)
- Deux corpus utilisés : un extrait INA (français parlé) et un cours d’informatique de 1969 (ENA)
- Utilisation de la métrique **PCIS** basée sur la distance de Levenshtein

## 🧪 Modèles testés

| Propriétaires     | Open Source             |
|-------------------|-------------------------|
| GPT-4.1           | DeepSeek-v3-turbo       |
| GPT-4.1-mini      | LLaMA-4-Maverick / Scout       |
| GPT-3.5-turbo     | Gemma-3  |

## 🧾 Résultats clés

- Meilleures performances : **GPT-4.1-mini + few shot + role prompting + Température 0.05**
- Jusqu’à **75 %** des lignes corrigées efficacement

## 📦 Contenu

- `data/` : Corpus de transcriptions et résultats 
- `lib/` : Scripts de traitement 
- `notebooks/` : Code Python pour du traitement et de la visualisation


## 🛠️ Technologies

- Python 3.x
- OpenAI API / Novita

## 📄 Références

- Boros et al. (2024), *Post-Correction of Historical Text Transcripts with LLMs*

