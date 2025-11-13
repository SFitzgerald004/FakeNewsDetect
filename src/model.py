# This is the file for the model to both train and test the dataset on
# 80% of the data is used as training, the remaining 20% is used for testing

# Used to pass the data from data_load file to the model
import data_load

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# These are the two variables used are for both training and testing
# The X variable contains the news titles, this will be used regardless testing or training modes
# The Y variable contains the label variable, which indicates if the news article is real or fake
# Y value of 1 indicates real news, whereas Y value of 0 indicates fake news
# The Y variable will only be used during training mode
X = data_load.df['title']
Y = data_load.df['real']

# This is the function used to train and test the model itself
# 
# test_size dictates the percentage of the data to be used for testing purpose as a double. 
# In this case, 20% of the data is being used to train the model, so a value of 0.2 is being used to define this
# random_state is a shuffler, making the dataset shuffled each time before training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print output
print("-------------- TRAINING OUTPUT -------------- \n")
print(f"[TRAINING SAMPLES] - {len(X_train)}")
print(f"[TESTING SAMPLES] - {len(X_test)} \n")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features = 5000,        # Keeps vectorizer at 5000 words max
    stop_words = 'english'      # Removes common stop words
)

# Fit on the training data and transform
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Vectorizer Prints
print("-------------- TF-IDF VECTORIZATION -------------- \n")
print(f"[NUMBER OF FEATURES] - {X_train_tfidf.shape[1]}")

# ---------------------------------------------------------------

# Create classifier
clf = LogisticRegression(max_iter = 1000)

# Train the model
clf.fit(X_train_tfidf, Y_train)

# Predict
y_pred = clf.predict(X_test_tfidf)

# Evaluate results
# Classification Report
print("-------------- CLASSIFICATION REPORT -------------- \n")
print(classification_report(Y_test, y_pred))
# When viewing results, look at the 'f1-score' column along with the accuracy row, this will show the average correctly classified

# Confusion Matrix
print("-------------- CONFUSION MATRIX REPORT -------------- \n")
print(confusion_matrix(Y_test, y_pred))
# Report rows: 
# Correct Fake (0) [num_correct, num_wrong]
# Correct True (1) [num_wrong, num_correct]
# To find error%, divide value 1 by value 2 in the second row, * 10

# Predict Fake News
def predict_fake_news(article_text):
    # Input: Single string - News Text
    # Output: 'Real' or 'Fake'
    vect = vectorizer.transform([article_text])
    pred = clf.predict(vect)[0]
    # In the dataset, 0 is fake, 1 is real
    return "Fake" if pred == 0 else "Real"