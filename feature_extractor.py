import nltk
import numpy
import re
from nltk.util import ngrams
import operator

# extract configured set of features from list of text instances

# global variables to pass around data
source_texts = []
word_counts = []
tokenized_texts = []
tagged_texts = []
stemmed_texts = []


def preprocess():
    """ prepares source_texts for feature extraction; called by extract_features

    puts words in lower case, tokenizes and stems them, and removes rare words
    no args and no return because of use of global variables
    """
    # lower case, count words, tokenize, and tag
    global source_texts, word_counts, tokenized_texts, tagged_texts
    source_texts = [text.lower() for text in source_texts]
    word_counts = [len(text.split()) for text in source_texts]
    tokenized_texts = [nltk.word_tokenize(text) for text in source_texts]
    tagged_texts = [[tag[1] for tag in nltk.pos_tag(text)]
                    for text in tokenized_texts]

    # stem using standard nltk porter stemmer
    porter = nltk.PorterStemmer()
    global stemmed_texts
    # stemmed_texts = [[porter.stem(t) for t in tokens]
    #                 for tokens in tokenized_texts]
    # iterating instead of list comprehension to allow exception handling
    for tokens in tokenized_texts:
        stemmed_text = []
        for t in tokens:
            try:
                stemmed_text.extend([porter.stem(t)])
            except IndexError:
                stemmed_text.extend('')
        stemmed_texts.append(stemmed_text)

    # remove rare words
    # vocab = nltk.FreqDist(w for w in line for line in stemmed_texts)
    vocab = nltk.FreqDist(w for text in stemmed_texts for w in text)
    rare_words_list = [re.escape(word) for word in vocab.hapaxes()]
    rare_words_regex = re.compile(r'\b(%s)\b' % '|'.join(rare_words_list))
    stemmed_texts = [[rare_words_regex.sub('<RARE>', w) for w in text]
                     for text in stemmed_texts]
    # note: source_texts will be lower case, but only stemmed_texts will have
    # rare words removed


def bag_of_function_words():
    """ returns, for each nltk stop word, count per text in source_texts """
    bow = []
    for sw in nltk.corpus.stopwords.words('english'):
        counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, text))
                  for text in source_texts]
        counts = [counts[i] / word_counts[i] for i in range(0, len(counts))]
        bow.append(counts)
    return bow


def bag_of_ngrams(texts, n=1, m=None):
    """ returns counts of up to m overall most common ngrams for each given text

    determines the counts of all ngrams, orders them by sum of counts across
    texts and returns counts for up to m most common ones

    args:
        texts: list of texts as list of list of words (or tags etc)
        n: 1 for unigram (default), 2 for bigram, 3 for trigram etc.
        m: upper limit for number of features; if none, all are returned

    returns:
        list of list of most common ngram counts, m x len(texts)
    """
    # generate list of lists of nrams for all texts
    ngrammed_texts = [list(ngrams(text, n)) for text in texts]

    # count ngrams in dictionaries, one for each text, plus one for sums
    cnts = []
    cnt_sum = {}
    for text in ngrammed_texts:
        cnts.append({})
        i = len(cnts) - 1
        for ngram in text:
            cnts[i][ngram] = 1 + (cnts[i][ngram] if ngram in cnts[i] else 0)
            cnt_sum[ngram] = 1 + (cnt_sum[ngram] if ngram in cnt_sum else 0)

    # create list of lists of counts for each text for the most common ngrams
    # first, sort the ngrams by total counts
    cnt_sorted = sorted(cnt_sum.items(), key=operator.itemgetter(1),
                        reverse=True)
    # then, create the bag of ngrams (up to m), normalized by word count
    bon = []
    for ngram, total in cnt_sorted:
        counts = [(cnt[ngram] if ngram in cnt else 0) for cnt in cnts]
        counts = [counts[i] / word_counts[i] for i in range(0, len(counts))]
        bon.append(counts)
        if m and len(bon) >= m:
            break
    return bon


def unique_words_ratio():
    """ returns #unique words / #words for each text

    uses stemmed words so 'eat' and 'eating' etc. are not treated as distinct
    (assuming they are stemmed correctly; 'eat' and 'ate' are still 'distinct');
    note that punctuation characters, parentheses etc. are treated as words
    """
    return [len(set(text)) / len(text) for text in stemmed_texts]


def words_per_sentence():
    """ returns average number of words per sentence for each text

    uses the '.' POS tag to detect number of sentences to avoid treating '.' in
    abbreviations as sentence ends
    """
    return [word_counts[i] / tagged_texts[i].count('.')
            for i in range(0, len(word_counts))]


def characters_per_words():
    """ returns average number of characters per word for each text

    note that character count includes punctuation, parentheses etc.
    """
    return [(len(source_texts[i]) - word_counts[i] + 1) / word_counts[i]
            for i in range(0, len(word_counts))]


# FILL IN OTHER FEATURE EXTRACTORS
# TODO topic models


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

    # extract requested features: TODO topic models
    if 'bag_of_function_words' in conf or all_features:
        features.extend(bag_of_function_words())
    if 'bag_of_pos_trigrams' in conf or all_features:
        features.extend(bag_of_ngrams(tagged_texts, 3, 500))
    if 'bag_of_pos_bigrams' in conf or all_features:
        features.extend(bag_of_ngrams(tagged_texts, 2, 100))
    if 'bag_of_pos_unigrams' in conf or all_features:
        features.extend(bag_of_ngrams(tagged_texts, 1, None))
    if 'bag_of_trigrams' in conf or all_features:
        features.extend(bag_of_ngrams(stemmed_texts, 3, 500))
    if 'bag_of_bigrams' in conf or all_features:
        features.extend(bag_of_ngrams(stemmed_texts, 2, 100))
    if 'bag_of_unigrams' in conf or all_features:
        features.extend(bag_of_ngrams(stemmed_texts, 1, None))
    if 'characters_per_word' in conf or all_features:
        features.extend(characters_per_words())
    if 'unique_words_ratio' in conf or all_features:
        features.extend(unique_words_ratio())
    if 'words_per_sentence' in conf or all_features:
        features.extend(words_per_sentence())

    # transpose list of lists so its dimensions are #instances x #features
    return numpy.asarray(features).T.tolist()
