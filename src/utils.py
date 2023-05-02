


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# preprocessing texts
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub('[^a-zA-Z \n\.]', '', text)

    # Tokenize into words
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # remove punctuation
    clean_words = [token.translate(str.maketrans('', '', string.punctuation)) for token in words]


    # Stem words
    stemmer = PorterStemmer()
    words = set([stemmer.stem(word) for word in clean_words if len(word)>1 and len(word)<12])

    return list(words)

