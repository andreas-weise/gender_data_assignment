import nltk
import numpy
import re

# extract configured set of features from list of text instances

# global variables to pass around data
source_texts = []
stemmed_texts = []


def preprocess():
    """ prepares source_texts for feature extraction; called by extract_features

    puts words in lower case, tokenizes and stems them, and removes rare words
    no args and no return because of use of global variables
    """
    # lower case and tokenize
    global source_texts
    source_texts = [text.lower() for text in source_texts]
    tokenized_texts = [nltk.word_tokenize(text) for text in source_texts]

    # stem using standard nltk porter stemmer
    porter = nltk.PorterStemmer()
    global stemmed_texts
    # stemmed_texts = [[porter.stem(t) for t in tokens]
    #                 for tokens in tokenized_texts]
    # iterating instead of list comprehension to allow exception handling
    for tokens in tokenized_texts:
        stemmed_line = []
        for t in tokens:
            try:
                stemmed_line.extend(porter.stem(t))
            except IndexError:
                stemmed_line.extend('')
        stemmed_texts.append(stemmed_line)

    # remove rare words
    # vocab = nltk.FreqDist(w for w in line for line in stemmed_texts)
    vocab = nltk.FreqDist(w for line in stemmed_texts for w in line)
    rare_words_list = vocab.hapaxes()
    rare_words_regex = re.compile(r'\b(%s)\b' % '|'.join(rare_words_list))
    stemmed_texts = [[rare_words_regex.sub('<RARE>', w) for w in line]
                     for line in stemmed_texts]
    # note: source_texts will be lower case, but only stemmed_texts will have
    # rare words removed


def bag_of_function_words():
    """ returns, for each nltk stop word, count per text in source_texts """
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, text))
                  for text in source_texts]
        bow.append(counts)
    return bow


# FILL IN OTHER FEATURE EXTRACTORS


def extract_features(texts, conf):
    """ extracts features in given conf from each text in given list of texts

    args:
        text: list of texts (essays) from which to extract features
        conf: set of identifiers of features to be extracted; from conf file

    returns:
        list of lists, #instances x #features = len(texts) x len(conf)
    """
    all_features = len(conf) == 0

    # use global variables to pass around data
    global source_texts
    source_texts = texts

    preprocess()

    # features will be list of lists
    # each component list will have the same length as the list of input text
    features = []

    # extract requested features: FILL IN HERE
    if 'bag_of_function_words' in conf or all_features:
        features.extend(bag_of_function_words())

    # transpose list of lists so its dimensions are #instances x #features
    features = numpy.asarray(features).T.tolist()

    return features
