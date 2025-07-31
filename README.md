# 💬 Sentiment Analysis: Comparing Machine Learning, Deep Learning, and Transformers

This project compares **three major NLP modeling approaches** for binary sentiment classification (positive vs. negative) using customer reviews on city passes:

1. **Classical Machine Learning (TF-IDF + XGBoost)**
2. **Deep Learning (BiLSTM + GloVe)**
3. **Transformers (BERT and ELECTRA)**

Each model was trained on the **same training set** and evaluated on the **same untouched test set**, with augmentation applied only to training data (where relevant). This ensures a clean comparison of model performance.

---

## 📁 Files Overview

- `XGBOOST.ipynb` → Classical ML pipeline (TF-IDF + XGBoost)
- `LSTM.ipynb` → Deep Learning model (BiLSTM + GloVe)
- `BERT.ipynb` → Transformer model (BERT fine-tuning)
- `ELECTRA.ipynb` → Transformer model (ELECTRA fine-tuning)
- `Augumented_Data.ipynb` → Back-translation logic (EN→DE→EN) using MarianMT

---

## 🧮 1. Machine Learning

### 🔹 TF-IDF + XGBoost

#### 📦 Pipeline

1. **Text Cleaning**: Lowercasing, punctuation removal, basic normalization.
2. **TF-IDF Vectorization**: Converts text into sparse matrix of n-gram frequencies (unigrams & bigrams).
3. **Data Augmentation**:
   - Applied only on training set.
   - Introduced minor text perturbations: character swaps, punctuation variation, typos.
   - Purpose: Mimic natural user-generated noise.
4. **Model**: XGBoost classifier with default tree-based boosting.
5. **Evaluation**: Predictions made on the untouched test set, followed by metric calculation.

#### ✅ Highlights
- Fast training and prediction.
- Performs surprisingly well with good feature engineering.
- Lacks deep context — each word is independent in the TF-IDF space.

---

## 🔁 2. Deep Learning

### 🔹 BiLSTM + GloVe

#### 📦 Pipeline

1. **Text Preprocessing**: Lowercase, remove stopwords, tokenize.
2. **Embedding**:
   - Used **pretrained GloVe embeddings (100 dimensions)**.
   - Each word in the sequence is converted to a fixed vector.
3. **Sequence Preparation**:
   - Tokenized texts were padded to equal length.
   - Maximum sequence length chosen based on percentile distribution.
4. **Model Architecture**:
   - **Embedding Layer** (with pretrained GloVe, frozen or trainable).
   - **BiLSTM Layer** with dropout for regularization.
   - **Dense Output Layer** with sigmoid activation for binary classification.
5. **Training**:
   - Only on the **cleaned, original training set** (no augmentation).
   - Used Binary Crossentropy loss, Adam optimizer.
6. **Evaluation**:
   - Same untouched test set used as above.
   - Predictions thresholded at 0.5 for classification metrics.

#### ✅ Highlights
- Captures word order and sequence dynamics.
- More robust than traditional ML, even without augmentation.
- Limitations: GloVe embeddings are **static** and cannot adapt to context.

---

## 🤖 3. Transformers

Transformer models use **self-attention mechanisms** to dynamically learn context-dependent representations of tokens. Both BERT and ELECTRA were fine-tuned on **augmented training data**.

### 🔹 Augmented Data (Back-Translation)

#### 📦 Method

- **Technique**: Back-translation using MarianMT.
- **Steps**:
  1. Translate English → German.
  2. Translate German → English.
- **Goal**: Introduce lexical diversity while preserving meaning.
- **Scope**: Applied **only on training data**.
- **Purpose**: Expose the model to diverse phrasings for better generalization.

---

### 🔹 BERT

#### 📦 Pipeline

1. **Tokenizer**: `bert-base-uncased` tokenizer applied to augmented training set.
2. **Data Formatting**:
   - Token IDs, attention masks generated.
   - Max length capped (e.g., 128 tokens).
3. **Fine-Tuning**:
   - `bert-base-uncased` model with a classification head (Dense layer).
   - Loss: Binary Crossentropy.
   - Optimizer: AdamW with linear learning rate decay.
   - Epochs: Typically 3–4 with early stopping.
4. **Evaluation**:
   - Inference on the **same untouched test set**.
   - Predictions processed using sigmoid activation and 0.5 threshold.

#### ✅ Highlights
- Strong contextual understanding from bidirectional attention.
- Handled lexically diverse inputs well, thanks to semantic augmentation.
- Achieved nearly perfect classification.

---

### 🔹 ELECTRA

#### 📦 Pipeline

1. **Tokenizer**: Same pipeline as BERT, but with ELECTRA tokenizer.
2. **Architecture**:
   - Uses a **discriminator-style pretraining** — predicts whether a token was replaced or not.
   - More compute-efficient, faster convergence than BERT.
3. **Fine-Tuning**:
   - Applied to **same back-translated training data**.
   - Classifier head added on top of `electra-small-discriminator`.
   - Used fewer epochs and still matched BERT in performance.
4. **Evaluation**:
   - Same test set as all other models.
   - Same metrics and thresholding strategy used.

#### ✅ Highlights
- Comparable accuracy to BERT.
- More efficient for low-resource or faster deployment scenarios.
- Great generalization on semantically augmented inputs.

---

## 📊 Evaluation

### 📈 Results Summary

| Model           | Accuracy | F1 Score | ROC AUC | PR AUC |
|----------------|----------|----------|---------|--------|
| TF-IDF + XGBoost | 0.9500  | 0.9500   | 0.9924  | 0.9952 |
| BiLSTM (GloVe)   | 0.9741  | 0.9700   | 0.9969  | 0.9981 |
| BERT             | 0.9900  | 0.9900   | **0.9999** | **1.0000** |
| ELECTRA          | 0.9900  | 0.9900   | 0.9998  | 0.9999 |

### 🔍 Interpretation

- **XGBoost**: Strong baseline with simple augmentation, but no contextual understanding.
- **BiLSTM**: Benefits from sequence modeling but lacks dynamic embeddings.
- **BERT & ELECTRA**: Vastly superior with deep semantic understanding.
- **ELECTRA**: Achieved BERT-like performance with faster training — practical edge.

---

## 🏁 Conclusion

This project showcases how **NLP modeling has evolved** over time, and how **architecture, embeddings, and augmentation techniques** collectively influence performance.

By keeping the test set fixed and using separate augmentation strategies per model, we ensure a fair and meaningful comparison. BERT and ELECTRA set the benchmark for modern NLP classification tasks.

---

## 🤝 Connect
For questions, collaborations, or opportunities related to AI in finance and risk modeling, feel free to connect through dhakshashganesan@gmail.com or GitHub.


