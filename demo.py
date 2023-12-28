import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC

# Load the pre-trained model
mymodel = pickle.load(open('model.pkl','rb'))  # Update with the correct path
X_train = np.load('X_train.npy')

D = 6 # Number of features
# download_dir = "/home/mhd/nltk_data"
# nltk.download('stopwords',download_dir=download_dir)
SW = nltk.corpus.stopwords.words("english")
PUNCT = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

# Standardize the features such that all features contribute equally to the distance metric computation of the SVM
scaler = StandardScaler()

# Fit only on the training data (i.e. compute mean and std)
scaler = scaler.fit(X_train)

# Vectorize function

def vectorize(w, scaled_position, p):
    # w : str : a token

    v = np.zeros(D).astype(np.float32)

    # If first character in uppercase
    if w[0].isupper():
        title = 1
    else:
        title = 0

    # All characters in uppercase
    if w.isupper():
        allcaps = 1
    else:
        allcaps = 0

    # Is stopword
    if w.lower() in SW:
        sw = 1
    else:
        sw = 0

    # Is punctuation
    if w in PUNCT:
        punct = 1
    else:
        punct = 0
    
    # is a proper noun(NNP/NNPS)
    if p == 22 or p == 23:
        pnoun = 1
    else:
        pnoun = 0

    # Build vector
    v[0] = title
    v[1] = allcaps
    #v[2] = len(w)
    v[2] = sw
    v[3] = punct
    v[4] = scaled_position
    v[5] = pnoun

    return v


def infer(model, scaler, s):
    # s: sentence

    tokens = word_tokenize(s)
    features = []
    postag = pos_tag(tokens)

    l = len(tokens)
    for i in range(l):
        pos = postag[i][1]
        if pos == "NNP":
          pos = 22
        elif pos == "NNPS":
          pos = 23
        f = vectorize(w = tokens[i], scaled_position = (i/l), p = pos)
        features.append(f)

    features = np.asarray(features, dtype = np.float32)

    scaled = scaler.transform(features)

    pred = model.predict(scaled)

    return pred, tokens, features



def main():
    st.title("Named Entity Recognition with SVM")

    # Input text box for user to enter a sentence
    user_input = st.text_input("Enter a sentence:")

    if st.button("Predict"):
        if user_input:
            # Call the predict_entities function
            pred,tokens,features = infer(mymodel, scaler, user_input)
            result = []
            for w, p in zip(tokens, pred):
                result.append(f"{w}_{int(p)}")
            result = " ".join(result)
            st.success(f"Predicted Named Entity: {result}")
        else:
            st.warning("Please enter a sentence.")

if __name__ == "__main__":
    main()
