import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    '/Users/MarcPlunkett/Desktop/datascience/nlp/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
N = dataset.shape[0]
clean_words = []

for n in range(0, N):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][n])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)

    clean_words.append(review)
clean_words
# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(clean_words).toarray()
y = dataset.iloc[:, 1].values

print(X)
# Splitting dataset into training nset and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_pred
cm
