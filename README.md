# 🔬 The Model Interrogator: A Comparative Study of NLP Robustness
## 🚀 [Click Here for the LIVE "Lite" Demo!](https://the-model-interrogator.streamlit.app)
    
**Note:** The live demo runs a "Lite" version (Models 1 & 2) due to Streamlit Cloud's 1GB memory limit. This full GitHub repository contains all the code, analysis, and saved model files for the complete 5-model gauntlet.

## Overview

This project tackles the **real-world challenge of automatically classifying public grievances**. While many NLP models achieve high accuracy, this project shows that **accuracy is a deceptive and insufficient metric** for real-world performance.

To prove this, we built **The Model Interrogator** — an advanced, interactive **Streamlit application** that stress-tests five NLP models under adversarial attacks.

Our definitive finding:
> The "classic" Logistic Regression + TF-IDF and the "modern" DistilBERT achieved a ~92% F1-score tie on clean data.  
> But under adversarial attack, the classic model was exposed as a *glass cannon*, while DistilBERT proved far more robust.

---

## Live Demo

The `app.py` lets you **interactively interrogate** all five models in real-time:
- Select from multiple attack types
- Adjust attack intensity
- Compare how each model’s predictions and confidence *flip* under stress

> 🎥



https://github.com/user-attachments/assets/bbd26ecd-8fc5-411e-b684-842eb2f2cdf5



---

## Key Application Features

### 5-Model Gauntlet
Compare five models in real-time:
1. Naive Bayes + TF-IDF  
2. Logistic Regression + TF-IDF  
3. Logistic Regression + Word2Vec  
4. LSTM + Word2Vec  
5. DistilBERT (Transformer)

### Expanded Attack Arsenal
Choose from multiple `textattack` transformations:
- **Semantic Attacks:** `WordSwapEmbedding`, `WordSwapWordNet`, `BERTMaskedLM`
- **Typo Attacks:** `WordSwapQWERTY`, `CharDelete`
- **Attack Intensity Slider:** Control aggression dynamically

### Real-Time Robustness Score
A live-calculated metric quantifying each model’s resilience based on:
- Prediction stability  
- Confidence retention  
- Semantic similarity

### Ensemble Voting System
A "committee" system combining predictions via:
- Majority Vote  
- Weighted Confidence  
- Highest Confidence

### Advanced Adversarial Metrics
- **Models Fooled:** How many predictions flipped (x/5)  
- **Semantic Similarity:** SentenceTransformer cosine score  
- **Average Confidence Drop:** Confidence erosion per model

### Interactive Visualizations
- **Text Diff Viewer:** Highlights deleted vs. inserted text  
- **Confidence Chart:** Plotly bar chart of before/after confidence  
- **Probability Heatmaps:** Seaborn visual comparison of distributions  
- **LIME Explanations:** Keyword highlights explaining Model 2 (LogReg + TF-IDF)

---

##  1. Dataset Source & Preprocessing

- **Source:** CFPB Consumer Complaint Database (7GB+) via [data.gov.in](https://data.gov.in)
- **Sample:** 1% random sample (~34,454 complaints)
- **Features:**
  - `Consumer complaint narrative` → `X`
  - `Product` → `y`

###  Cleaning & Merging
Merged redundant top classes:
- `Credit reporting or other personal consumer reports`  
- `Credit reporting, credit repair services, or other personal consumer reports`  
→ Combined as **Credit Reporting**

### Filtering
Kept top 4 distinct product categories → final **29,359 complaints** (imbalanced).

<img width="1034" height="450" alt="newplot" src="https://github.com/user-attachments/assets/61af8f97-df09-4484-ad56-cb04ce665a25" />


---

## 2. Methods: The 5-Model Gauntlet

| Model | Technique | Hypothesis |
|--------|------------|-------------|
| **1. Naive Bayes + TF-IDF** | “The Strawman” | Fast keyword counter baseline |
| **2. LogReg + TF-IDF** | “The Classic Hero” | Keyword weighting > simple counting |
| **3. LogReg + Word2Vec** | “Semantic-but-Dumb” | Semantics > keyword frequency |
| **4. LSTM + Word2Vec** | “The Sequential Thinker” | Sequence + semantics = improvement |
| **5. DistilBERT** | “The Modern Champion” | Deep bi-directional context wins |

---

##  3. How to Run This Project

###  Clone the Repository
```bash
git clone https://github.com/AmeenKhan12345/The-Model-Interrogator.git
cd The-Model-Interrogator
```
##  Install Dependencies

> 💡 **Tip:** Use a virtual environment.  
> DistilBERT requires `torch` with **CUDA** for faster performance.

```bash
pip install -r requirements.txt
```
##  Run the Interactive App

```bash
streamlit run app1.py
```
> (Pre-trained models, tokenizers, and encoders are included and auto-loaded.)

##  (Optional) Re-run Robustness Evaluation 
```bash
python evaluate_robustness.py
```
##  4. Experiments & Results

###  Experiment 1: The “Accuracy Trap”

| Model | F1-Score (Weighted) | F1-Score (Macro) | Log-Loss |
|--------|----------------------|------------------|-----------|
| Naive Bayes | 0.77 | 0.51 | 1.02 |
| LogReg + TF-IDF | 0.92 | 0.87 | 0.25 |
| LogReg + W2V | 0.90 | 0.84 | 0.32 |
| LSTM + W2V | 0.92 | 0.87 | 0.27 |
| DistilBERT | 0.93 | 0.88 | 0.22 |
###  Key Findings

- **Finding 1:** Models 2, 4, and 5 tied on clean data → **accuracy alone is misleading.**
- **Finding 2:** *Macro F1 < Weighted F1* → **models struggle with minority classes.**
- **Finding 3:** **DistilBERT and LogReg** are the most confident models *(lowest Log-Loss).*
###  Experiment 2: Confusion Matrix Analysis

> *(Run `calculate_metrics.py` to generate confusion matrices and add images — `cm_model_X.png`.)*

| Model | Observation |
|--------|--------------|
| 2. LogReg + TF-IDF | Shows a **clean diagonal** but confuses *Debt Collection* vs *Credit Reporting* — later exploited by our **Keyword Killer** attack. |
| 4. LSTM + W2V | **Stable across classes**, slightly weaker on rare categories. |
| 5. DistilBERT | **Strong class separation**, minimal cross-category confusion. |

** Finding:**  
Models **2**, **4**, and **5** show cleaner diagonals → better prediction stability.  
However, **Model 2** consistently misclassifies *Debt Collection* ↔ *Credit Reporting*, revealing a **semantic brittleness** later tested in adversarial experiments.
###  Experiment 3: The “Robustness Tie-Breaker”

A **Synonym Replacement Attack** was performed on **200 balanced samples (50 per class)** to test robustness.  
We measured **True Flip Rate (↓)** — the percentage of *correct predictions that flipped* under attack.

| Model | True Flip Rate ↓ | Flipped / Correct |
|--------|------------------|-------------------|
| LogReg + TF-IDF | 24.26% | (41 / 169) |
| LSTM + W2V | 16.67% | (28 / 168) |
| LogReg + W2V | 14.20% | (24 / 169) |
| Naive Bayes | 10.98% | (9 / 82) |
| DistilBERT | **6.98%** | (12 / 172) |

---

###  Key Finding

**DistilBERT** is the clear **robustness champion** — **3.5× more resilient** than LogReg + TF-IDF.  
The “classic” model’s accuracy was largely due to **memorized keywords**, not true semantic understanding.
## 5. Conclusion: What We Learned

- **Accuracy ≠ Robustness:** High test accuracy can **hide brittleness**.  
- **Context is King:** Models with **semantic understanding** (W2V, LSTM, BERT) show **higher resilience**.  
- **Robustness > Accuracy:** LogReg (92%) vs. DistilBERT (93%) were tied in accuracy, but robustness (24% vs. 7% flip rate) **reveals the truth**.  
- **Committee Models Win:** **Ensembles** provide **stability** against adversarial noise.

> ⚠️ **For real-world systems**, robustness testing and ensemble validation should be **mandatory before deployment.**
##  6. References

- **Devlin, J. et al. (2018).** [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*](https://arxiv.org/abs/1810.04805)

- **Morris, J. et al. (2020).** [*TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Model Training in NLP.*](https://arxiv.org/abs/2005.05909)

- **Ribeiro, M. et al. (2016).** [*"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME Paper).*](https://arxiv.org/abs/1602.04938)

- **CFPB (2023).** [*Consumer Complaint Database.*](https://data.gov.in/)

- **Reimers, N., & Gurevych, I. (2019).** [*Sentence-Transformers Library.*](https://www.sbert.net/)

---

💡 *“The Model Interrogator” reveals that accuracy is only the start of the story.  True intelligence lies in **robustness** — how models behave when the world gets noisy.*

