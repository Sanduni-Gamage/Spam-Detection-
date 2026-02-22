# 📧 Email Spam Detection - NLP Classification Project

A machine learning project that classifies emails as **Spam** or **Ham** (not spam) using Natural Language Processing (NLP) techniques and multiple classifiers.

## 📌 Overview

The project builds a complete text classification pipeline - from raw email text to a trained model that can predict whether any new email is spam. It compares multiple models and selects the best performer.

## 📂 Project Structure

```
Spam Detection/
├── Datasets/
│   └── Spam_Ham_Dataset.csv       # 5,508 labeled emails
├── spam_detection.ipynb           # Full analysis notebook
└── README.md
```

## 🔄 Pipeline

| Step                  | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| 1. Load & Explore     | Load 5,508 emails (28.2% spam, 71.8% ham), inspect samples                 |
| 2. Text Preprocessing | Lowercase, remove URLs/emails/punctuation/numbers, strip whitespace        |
| 3. Feature Extraction | TF-IDF vectorization (5,000 features), 80/20 train-test split              |
| 4. Model Training     | Train and compare Naive Bayes, Logistic Regression, Random Forest          |
| 5. Evaluation         | Confusion matrix, classification report, ROC curve, precision-recall       |
| 6. Prediction         | `check_email()` function for classifying new emails with confidence scores |

## 🔑 Key Results

| Model               | Accuracy  |
| ------------------- | --------- |
| **Naive Bayes**     | **97.4%** |
| Random Forest       | 97.4%     |
| Logistic Regression | 96.3%     |

**Best Model: Multinomial Naive Bayes** — selected for its speed and top accuracy.

```
Classification Report:
              precision    recall  f1-score
    Ham          0.97      1.00      0.98
    Spam         0.99      0.92      0.95
    Accuracy                         0.97
```

## 📊 Visualizations

- Confusion Matrix
- ROC Curve with AUC score
- Precision-Recall Curve
- Model Accuracy Comparison (bar chart)
- Spam vs Ham distribution
- Text length analysis
- Top predictive words for spam and ham

## 🚀 Usage

```python
# After running the notebook, use the prediction function:
check_email("FREE MONEY! Win $10000 now! Click here immediately!")
# → Prediction: Spam | Confidence: 99.8%

check_email("Meeting scheduled for tomorrow at 2 PM in conference room B")
# → Prediction: Ham | Confidence: 99.9%
```

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/GamithaManawadu/Email-Spam-Detection-with-Real-Dataset.git
cd email-spam-detection-with-real-dataset

# Install dependencies
pip install -r requirements.txt
```

## 🛠 Tech Stack

- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualization
- **scikit-learn** — TF-IDF vectorization, model training, evaluation
- **re** — regex-based text cleaning
