# sentiment_analysis_ML-project
                 ----this project was given by my ML faculty of my college


# Sentiment Analysis on Twitter Training Dataset  
**Machine Learning Project (Non–Deep Learning Approach)**  
Algorithms Used: Logistic Regression, Naive Bayes, KNN  

---

## 📌 Project Overview
This project performs sentiment analysis on a large Twitter dataset (`twitter_training.csv`) using **classical machine learning algorithms**.  
Deep learning is intentionally avoided to stay aligned with academic syllabus constraints.

The goal is to:
- Classify tweets into **Positive / Negative / Neutral**
- Compare multiple ML models
- Plot evaluation graphs
- Identify the **best-performing algorithm**
- Provide analysis and insights based on performance metrics

---

## 📁 Dataset Format (`twitter_training.csv`)
 CSV file  follow the format:


- `tweet_id` → Numeric ID 
- `game` → Category (e.g., Borderlands)  
- `label` → Sentiment (Positive, Negative, Neutral)  
- `text` → Tweet content  

Dataset size: **~75,000 rows**

---

## ⚙️ Machine Learning Algorithms Used
The project uses only algorithms allowed in your course:

### **1. Logistic Regression**
- Acts as the primary classifier  
- Works best with TF-IDF text features  
- Typically highest F1-score  

### **2. Naive Bayes (MultinomialNB)**
- Fast and efficient  
- Works well for word-frequency based text  

### **3. K-Nearest Neighbors (KNN)**
- Included for comparison  
- Performs weaker due to high-dimensional TF-IDF space  

---

## 📊 Evaluation Metrics
For every model, the following are computed:

- **Accuracy**
- **Precision (Macro)**
- **Recall (Macro)**
- **F1 Score (Macro)**
- **Confusion Matrix**
- **ROC Curve (if supported)**

---

## 📈 Generated Plots
The script produces the following:

| File | Description |
|------|-------------|
| `confusion.png` | Confusion matrix for the best model |
| `roc.png` | ROC Curve (Positive vs Rest) |
| `comparison_metrics.png` | Comparative bar charts of Accuracy, Precision, Recall, F1 for all models |

---

## 📄 Output Reports
The script saves:

### **1. metrics_report.csv**  
Contains performance metrics for all three models.

Example structure:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|

### **2. Console Observations**
A detailed summary is printed, including:
- Model performance comparison  
- Analysis of confusion matrix  
- Insight from ROC curves  
- Explanation of trends and errors  

---

## 🚀 How to Run the Project

### **1. Install Required Libraries**
```bash
pip install pandas numpy scikit-learn matplotlib
