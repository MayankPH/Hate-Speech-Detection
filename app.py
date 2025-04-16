# Importing Necessary Libraries
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Feature Selection
from sklearn.model_selection import train_test_split

# Model Selection
from sklearn.tree import DecisionTreeClassifier

# Data Collection
data = pd.read_csv("twitter.csv")

# Adding a new column to the dataset as labels
data["labels"] = data["class"].map({0: "Hate Speech",
                                    1: "Offensive Language",
                                    2: "No Hate"})

# Selecting the necessary columns
data = data[["tweet", "labels"]]
print(data.head())
print(data.columns)
print(data['labels'].value_counts())

# Importing required libraries for text processing
import re
import nltk
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

# Cleaning Data
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)  # Remove any text inside square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('\n', '', text)  # Remove newlines
    text = re.sub('\w*\d\w*', '', text)  # Remove words with numbers
    
    # Keep offensive words in the text for feature extraction
    stopword = set(stopwords.words('english'))
    text = [word for word in text.split(' ') if word not in stopword]
    
    # Use stemming or lemmatization (if needed) but avoid stemming offensive words like 'bastard'
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    
    return text


# Cleaning the 'tweet' column
data["tweet"] = data["tweet"].apply(clean)

# Feature Selection
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Model Selection
cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the data

# Splitting into Train and Test sets
X_train, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initializing the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Printing training and testing accuracy
print("Training accuracy:", clf.score(X_train, y_train))
print("Testing accuracy:", clf.score(Xtest, y_test))

# Streamlit App
def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    
    # Check if the user entered any text
    if len(user) < 1:
        st.write("Please enter a tweet to analyze.")
    else:
        # Preprocess the input and make the prediction
        sample = user
        data = cv.transform([sample]).toarray()
        prediction = clf.predict(data)
        
        # Map numeric prediction to labels
        prediction_label = prediction[0]  # Get the label
        st.write(f"Prediction: {prediction_label}")

        if prediction_label == "Hate Speech":
            st.write("This tweet contains hate speech.")
        elif prediction_label == "Offensive Language":
            st.write("This tweet contains offensive language.")
        else:
            st.write("This tweet does not contain hate or offensive language.")

# Running the app
hate_speech_detection()

# Hate Speech: "Kill all [ethnic group]!"
# Offensive Language: "You're such a Bitch!"
# No Hate: "Awwwwe! This is soooo ADORABLE!"
# streamlit run app.py
