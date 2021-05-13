from elasticsearch_dsl import (  # type: ignore
    Document,
    Text,
    Keyword,
    DenseVector,
    Date,
    token_filter,
    analyzer,
)

custom_analyzer = analyzer(
    "custom_analyzer",
    tokenizer="standard",
    filter=["asciifolding", "snowball", "lowercase"],
)


# Synonyms mapping
# my_synonyms = ["effort, urges, to free, released, free, release, immediate release, will release, frees, departure of Americans from Iran",
#                "Washington Post journalist, Jason Rezaian",
#                "Obama, U.N. rights committee, the State Department",
#                "soon, at last",
#                "detention, held, detained, arrest"]

# Add synonyms analyzer to build a new index
# Author: Shi Qiu
synonyms_token_filter = token_filter(
    'synonyms_token_filter',     # Any name for the filter
    'synonym',                   # Synonym filter type
    synonyms_path='pa5_data/synonym.txt'       # Synonyms mapping will be inlined
)

synonyms_analyzer = analyzer("synonyms_analzyer", tokenizer="standard", filter=[
                             "lowercase", synonyms_token_filter])


class BaseDoc(Document):
    """
    wapo document mapping structure
    """

    doc_id = (
        Keyword()
    )  # we want to treat the doc_id as a Keyword (its value won't be tokenized or normalized).
    title = (
        Text()
    )  # by default, Text field will be applied a standard analyzer at both index and search time
    author = Text()
    content = Text(
        analyzer="standard"
    )  # we can also set the standard analyzer explicitly
    custom_content = Text(
        analyzer=synonyms_analyzer
    )
    date = Date(
        format="yyyy/MM/dd"
    )  # Date field can be searched by special queries such as a range query.
    annotation = Text()
    # fasttext embedding in the DenseVector field
    ft_vector = DenseVector(dims=300)
    sbert_vector = DenseVector(
        dims=768
    )  # sentence BERT embedding in the DenseVector field

    def save(self, *args, **kwargs):
        """
        save an instance of this document mapping in the index
        this function is not called because we are doing bulk insertion to the index in the index.py
        """
        return super(BaseDoc, self).save(*args, **kwargs)


if __name__ == "__main__":
    pass
