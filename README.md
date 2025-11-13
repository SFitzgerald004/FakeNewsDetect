# FakeNewsDetect

A machine learning-based fake news detection system that uses Natural Language Processing (NLP) to classify news articles as real or fake based on their titles.

## Overview

This project implements a Logistic Regression classifier with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to detect fake news. The model is trained on the FakeNewsNet dataset, which contains news titles labeled as real (1) or fake (0).

## Features

- **TF-IDF Vectorization**: Converts news titles into numerical features with a maximum of 5000 features
- **Logistic Regression Model**: Uses scikit-learn's LogisticRegression classifier for binary classification
- **Comprehensive Evaluation**: Provides classification reports and confusion matrices
- **Prediction Function**: Allows classification of new news articles
- **Dataset Analysis**: Includes data loading and exploration utilities

## Dataset

The project uses the **FakeNewsNet.csv** dataset located in the `data/` directory. The dataset contains:

- **title**: The headline of the news article
- **news_url**: URL of the original article
- **source_domain**: Domain of the news source
- **tweet_num**: Number of tweets associated with the article
- **real**: Label indicating if the news is real (1) or fake (0)

The dataset contains **23,196 news articles** with approximately:
- 75.5% real news articles
- 24.5% fake news articles

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SFitzgerald004/FakeNewsDetect.git
cd FakeNewsDetect
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.x
- joblib==1.5.2
- numpy==2.3.4
- pandas==2.3.3
- python-dateutil==2.9.0.post0
- pytz==2025.2
- scikit-learn==1.7.2
- scipy==1.16.3
- six==1.17.0
- threadpoolctl==3.6.0
- tzdata==2025.2

## Usage

### Running the Model

To train and test the model, run:

```bash
cd src
python model.py
```

This will:
1. Load and analyze the dataset
2. Split data into training (80%) and testing (20%) sets
3. Vectorize the text using TF-IDF
4. Train the Logistic Regression model
5. Display evaluation metrics including:
   - Classification report (precision, recall, f1-score)
   - Confusion matrix
   - Model accuracy

### Loading Dataset Information

To view dataset structure and preview:

```bash
cd src
python data_load.py
```

### Using the Prediction Function

You can use the `predict_fake_news()` function in your code to classify individual articles:

```python
from model import predict_fake_news

# Example usage
article_title = "Breaking: Unbelievable discovery shocks scientists!"
result = predict_fake_news(article_title)
print(f"This article is: {result}")  # Output: "Fake" or "Real"
```

## Model Details

### Architecture

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Max features: 5000
  - Stop words: English common words removed
  
- **Classifier**: Logistic Regression
  - Max iterations: 1000
  - Solver: Default (lbfgs)

### Training Process

1. **Data Split**: 80% training, 20% testing (random_state=42)
2. **Text Vectorization**: Converts text to numerical features using TF-IDF
3. **Model Training**: Trains on vectorized training data
4. **Evaluation**: Tests on held-out test set

## Model Performance

Typical performance metrics:
- **Overall Accuracy**: ~83%
- **Precision (Real News)**: ~84%
- **Recall (Real News)**: ~95%
- **F1-Score (Real News)**: ~89%

Note: The model performs better at identifying real news than fake news due to the class imbalance in the dataset.

## Understanding the Output

### Classification Report
```
              precision    recall  f1-score   support
           0       0.76      0.44      0.56      1131
           1       0.84      0.95      0.89      3509
```
- **Class 0**: Fake news
- **Class 1**: Real news
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-score**: Harmonic mean of precision and recall

### Confusion Matrix
```
[[ 499  632]
 [ 160 3349]]
```
- Row 1: Fake news predictions [correct, incorrect]
- Row 2: Real news predictions [incorrect, correct]

## Project Structure

```
FakeNewsDetect/
│
├── data/
│   └── FakeNewsNet.csv          # Dataset file
│
├── src/
│   ├── data_load.py             # Data loading and exploration
│   └── model.py                 # Model training, testing, and prediction
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Future Improvements

Potential enhancements for the project:
- Implement additional classifiers (Random Forest, SVM, Neural Networks)
- Use more text features (article body, metadata)
- Apply deep learning models (LSTM, BERT)
- Handle class imbalance with techniques like SMOTE
- Create a web interface for real-time predictions
- Add cross-validation for robust model evaluation

## License

This project is available for educational and research purposes.

## Author

SFitzgerald004

## Acknowledgments

- FakeNewsNet dataset providers
- scikit-learn library contributors
