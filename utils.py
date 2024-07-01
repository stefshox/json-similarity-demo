import re
import math
from collections import defaultdict


def get_app_search_value(app_data):
    """Get search value for an app.

    Parameters
    ----------
    app_data : dict
        Dictionary representing the app data

    Returns
    -------
    str
        Search value for the app
    """

    search_values = []
    for table in app_data["tables"]:
        fields = ", ".join(field["name"] for field in table["fields"])
        search_values.append(f"{table['name']} ({fields})")
    return ", ".join(search_values)

def preprocess_text(text):
    """Tokenize and preprocess text.
    
    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    list
        List of preprocessed words
    """

    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def calculate_tf(doc):
    """Calculate term frequency for a document.
    
    TF(word) = (Number of times word appears in a document) / (Total number of words in the document)
    
    Parameters
    ----------
    doc : list
        List of words in the document

    Returns
    -------
    tf : dict
        A dictionary with word as key and its TF value as value
    """

    tf = defaultdict(int)
    for word in doc:
        tf[word] += 1
    doc_length = len(doc)
    for word in tf:
        tf[word] /= doc_length
    return tf

def calculate_idf(docs):
    """Calculate inverse document frequency for all documents.
    
    IDF(word) = log(Total number of documents / (Number of documents containing the word))
    
    Parameters
    ----------
    docs : list
        List of documents where each document is a list of words

    Returns
    -------
    idf : dict
        A dictionary with word as key and its IDF value as value
    """

    idf = defaultdict(int)
    total_docs = len(docs)
    for doc in docs:
        unique_words = set(doc)
        for word in unique_words:
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(total_docs / (1 + idf[word]))
    return idf

def calculate_tfidf(tf, idf):
    """Calculate TF-IDF for a document.
    
    TF-IDF(word) = TF(word) * IDF(word)
    
    Parameters
    ----------
    tf : dict
        Dictionary of word and its TF value
    idf : dict
        Dictionary of word and its IDF value

    Returns
    -------
    tfidf : dict
        A dictionary with word as key and its TF-IDF value as value
    """
    
    tfidf = {}
    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf.get(word, 0)
    return tfidf

def dot_product(vec1, vec2):
    """Calculate the dot product of two vectors.

    Parameters
    ----------
    vec1 : dict
        Dictionary representing a vector
    vec2 : dict
        Dictionary representing a vector

    Returns
    -------
    dot_product_value : float
        Dot product of the two vectors
    """
    
    dot_product_value = 0.0
    for key in vec1:
        if key in vec2:
            dot_product_value += vec1[key] * vec2[key]
    return dot_product_value

def magnitude(vec):
    """Calculate the magnitude of a vector.

    Parameters
    ----------
    vec : dict
        Dictionary representing a vector

    Returns
    -------
    vec_magnitude : float
        Magnitude of the vector
    """
    
    sum_of_squares = sum([value**2 for value in vec.values()])
    magnitude = math.sqrt(sum_of_squares)
    return magnitude

def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors.
    
    Parameters
    ----------
    vec1 : dict
        Dictionary representing a vector
    vec2 : dict
        Dictionary representing a vector

    Returns
    -------
    cos_similarity : float
        Cosine similarity between the two vectors
    """

    dot_prod = dot_product(vec1, vec2)
    magnitude1 = magnitude(vec1)
    magnitude2 = magnitude(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    cos_similarity = dot_prod / (magnitude1 * magnitude2)
    return cos_similarity