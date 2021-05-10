import string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


def wordnet_query_expansion(query, num_synonyms):
    query = query.replace('\n', '')
    word_tokens = word_tokenize(query)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    synonyms = []

    count = 0
    for x in filtered_sentence:
        for syn in wordnet.synsets(x):
            for l in syn.lemmas():
                if(count < num_synonyms):  # put three syn in the result
                    if l.name() not in synonyms:
                        synonyms.append(l.name())
                        count += 1
        count = 0

    synonyms_string = ' '.join(synonyms)
    return synonyms_string


if __name__ == "__main__":
    wordnet_query_expansion(
        "Washington Post journalist Jason Rezaian had served in an Iranian prison and was released, as well as those that describe efforts from the Washington Post and others to have him released.")
