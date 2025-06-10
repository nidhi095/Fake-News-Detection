# 📰 Fake News Detection using Machine Learning

This project uses machine learning and natural language processing (NLP) to detect fake news articles. It trains a **Logistic Regression** model on a dataset of real and fake news, using **TF-IDF vectorization** to convert text into machine-readable features.

---

## 🎯 Project Goals

- Build a machine learning pipeline to detect whether a news article is real or fake.
- Apply text preprocessing and vectorization using NLP techniques.
- Train and evaluate a classifier.
- Allow users to test the model on custom news inputs.

---

## 🧠 Key Concepts Used

| Concept | Description |
|--------|-------------|
| **Natural Language Processing (NLP)** | Techniques for cleaning and processing text data. |
| **TF-IDF (Term Frequency-Inverse Document Frequency)** | Converts raw text into meaningful numerical vectors. |
| **Logistic Regression** | A linear classifier for binary classification problems. |
| **Train/Test Split** | Divides data for training and evaluation. |
| **Model Evaluation** | Accuracy, Precision, Recall, and F1-Score to assess performance. |

---

## 📁 Dataset

We use the open-source dataset from Kaggle:  
👉 [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| File | Description |
|------|-------------|
| `Fake.csv` | Contains ~21,000 fake news articles |
| `True.csv` | Contains ~21,000 real news articles |

> ⚠️ Note: Due to GitHub file size limits, the dataset is not uploaded.  
> Please download both files from Kaggle and place them in the same directory as the Python script.

---

## 📂 Project Structure

