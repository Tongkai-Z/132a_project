from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

'''
This module provide method for expanding the query using wordnet synonyms

@author Tongkai Zhang
'''


def wordnet_query_expansion(query, num_synonyms):
    '''
    this method expand the query by adding synonyms to each term.
    '''
    word_tokens = word_tokenize(query)
    s = [w for w in word_tokens if not w in stop_words]
    synonyms = []
    for term in s:
        count = 0
        for syn in wordnet.synsets(term):
            for l in syn.lemmas():
                if(count < num_synonyms):  # put syn in the result
                    if l.name() not in synonyms:
                        synonyms.append(l.name())
                        count += 1
    res = ' '.join(synonyms)
    return res


if __name__ == "__main__":
    print(wordnet_query_expansion(
        "Jason Rezaian released from Iran", 3))
