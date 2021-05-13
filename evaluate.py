import argparse
from utils import parse_wapo_topics
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
from metrics import ndcg

INTERACTIVE_INDEX = "wapo_docs_50k"
INTERACTIVE_TOP = 20


def build_query(topic_id, query_type, customized_content):
    # load the topic as query string
    # 0:title, 1:description 2:narratives
    type_mapping = {"title": 0, "description": 1, "narration": 2}
    query_string = parse_wapo_topics(
        "pa5_data/topics2018.xml")[topic_id][type_mapping[query_type]]
    q_basic = None
    if customized_content:
        q_basic = Match(
            custom_content={"query": query_string}
        )
    else:
        q_basic = Match(
            content={"query": query_string}
        )
    return q_basic


def process_interactive_query(query_string, search_type):
    # search_type: bm_default bm_customized ft_vector sbert_vector
    q_basic = None
    if search_type == "bm_customized":
        q_basic = Match(
            custom_content={"query": query_string}
        )
    else:
        q_basic = Match(
            content={"query": query_string}
        )
    response = search(INTERACTIVE_INDEX, q_basic, INTERACTIVE_TOP)
    # embedding reranking
    # if search_type == "ft_vector" or search_type == "sbert_vector":
    if search_type == "ft_vector" or search_type.startswith("sbert_vector"):
        vector_mapping = {"sbert_vector": "sbert", "ft_vector": "fasttext", "sbert_dpr_vector": "sbert_dpr", "sbert_dot_product_vector": "sbert_dot_product"}
        result_list = [hit.meta.id for hit in response]
        q_match_ids = Ids(values=result_list)
        encoder = EmbeddingClient(
            host="localhost", embedding_type=vector_mapping[search_type])
        query_vector = encoder.encode([query_string], pooling="mean").tolist()[
            0
        ]
        q_vector = generate_script_score_query(
            query_vector, search_type
        )
        q_c = (q_match_ids & q_vector)
        response = search(INTERACTIVE_INDEX, q_c, INTERACTIVE_TOP)
    return response


def search(index, query, top_k):
    s = Search(using="default", index=index).query(query)[
        :top_k
    ]  # initialize a query and return top five results
    response = s.execute()
    return response


def print_response(response):
    for hit in response:
        print(
            hit.meta.id, hit.meta.score, hit.title, hit.annotation, sep="\t"
        )


def embedding_reranked(result_list, index_name, vector_name, topic_id, query_type, top_k):
    query = build_embedding_query(
        result_list, vector_name, topic_id, query_type)
    return search(index_name, query, top_k)


def build_embedding_query(result_list, vector_name, topic_id, query_type):
    type_mapping = {"title": 0, "description": 1, "narration": 2}
    vector_mapping = {"sbert_vector": "sbert", "ft_vector": "fasttext", "sbert_dpr_vector": "sbert_dpr", "sbert_dot_product_vector": "sbert_dot_product"}
    query_string = parse_wapo_topics(
        "pa5_data/topics2018.xml")[topic_id][type_mapping[query_type]]
    q_match_ids = Ids(values=result_list)
    encoder = EmbeddingClient(
        host="localhost", embedding_type=vector_mapping[vector_name])
    query_vector = encoder.encode([query_string], pooling="mean").tolist()[
        0
    ]
    q_vector = generate_script_score_query(
        query_vector, vector_name
    )
    q_c = (q_match_ids & q_vector)
    return q_c

def vector_map(vector_name):
    if vector_name=="ft_vector":
        return "ft_vector"
    elif vector_name.startswith("sbert_"):
        return "sbert_vector"


def generate_script_score_query(query_vector, vector_name):
    """
    generate an ES query that match all documents based on the cosine similarity
    :param query_vector: query embedding from the encoder
    :param vector_name: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")
    :return: an query object
    """
    q_script = ScriptScore(
        query={"match_all": {}},  # use a match-all query
        script={  # script your scoring function
            "source": f"cosineSimilarity(params.query_vector, '{vector_map(vector_name)}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return q_script


def generate_ndcg_score(topic_id, response):
    relevance_list = [parse_score(topic_id, hit.annotation)
                      for hit in response]
    return ndcg(relevance_list, len(relevance_list))


def parse_score(topic_id, annotation):
    if annotation == "" or annotation == None:
        return 0
    if annotation[:annotation.index("-")] != topic_id:
        return 0
    return int(annotation[annotation.index("-") + 1:])


if __name__ == "__main__":
    connections.create_connection(
        hosts=["localhost"], timeout=100, alias="default")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name", required=True, type=str, help="name of the index"
    )
    parser.add_argument(
        "--topic_id", required=True, type=str, help="topic id"
    )
    parser.add_argument(
        "--query_type",
        required=True,
        type=str,
        help="query type for the topic"
    )
    parser.add_argument(
        "--top_k",
        required=True,
        type=int,
        help="number of results returned"
    )
    parser.add_argument(
        "--vector_name",
        required=False,
        type=str,
        default="",
        help="embedding vector name if used"
    )
    parser.add_argument(
        "-u",
        dest='customized_content',
        action='store_true',
        help="use customized_content"
    )
    parser.set_defaults(customized_content=False)
    args = parser.parse_args()
    # build the query
    query = build_query(args.topic_id, args.query_type,
                        args.customized_content)
    # process the BM25 ranked search
    response = search(args.index_name, query, args.top_k)
    # reranked by embedding
    if args.vector_name != "":
        result_list = [hit.meta.id for hit in response]
        response = embedding_reranked(result_list,
                                      args.index_name, args.vector_name, args.topic_id, args.query_type, args.top_k)
    print(generate_ndcg_score(args.topic_id, response))
