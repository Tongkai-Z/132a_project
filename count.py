import argparse
from utils import parse_wapo_topics
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
from metrics import ndcg

INTERACTIVE_INDEX = "wapo_docs_50k"
INTERACTIVE_TOP = 20
relative_docs = {}
fn = []
fp = []


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


def build_count_query(topic_id):
    query_string = topic_id + "-"
    q_basic = Match(
        annotation={"query": query_string}
    )
    return q_basic


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
    vector_mapping = {"sbert_vector": "sbert", "ft_vector": "fasttext"}
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
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
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


def count(response):
    global relative_docs
    res = [0, 0, 0]
    for hit in response:
        if hit.annotation != "" and hit.annotation != None:
            rel = int(hit.annotation[hit.annotation.index("-") + 1:])
            if rel != 0:
                relative_docs[hit.meta.id] = hit
            res[rel] += 1
    return res


def get_fnfp(response, show_fpfn):
    global fn
    global fp
    tp_id = []
    res = [0, 0]
    fn_relative_map = {}
    for hit in response:
        if hit.meta.id not in relative_docs:
            res[0] += 1  # false positive
            fp.append(hit.title)
        else:
            tp_id.append(hit.meta.id)
    res[1] = len(relative_docs) - (len(response) - res[0])
    # relative - tp_id
    # check the relative score of the false negative docs
    for id in relative_docs:
        if id not in tp_id:
            fn.append(relative_docs[id].title)
            if relative_docs[id].annotation in fn_relative_map:
                fn_relative_map[relative_docs[id].annotation] = fn_relative_map[relative_docs[id].annotation] + 1
            else:
                fn_relative_map[relative_docs[id].annotation] = 1
    if show_fpfn:
        print("false positive:\n", fp)
        print("false negative:\n", fn)
        print("false negative relevance: ", fn_relative_map)
    return res


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
        "-fpfn",
        dest='show_fpfn',
        action='store_true',
        help="show the title of fpfn"
    )
    parser.add_argument(
        "-u",
        dest='customized_content',
        action='store_true',
        help="use customized_content"
    )
    parser.set_defaults(customized_content=False)
    parser.set_defaults(show_fpfn=False)
    args = parser.parse_args()
    # build the query
    query = build_query(args.topic_id, args.query_type,
                        args.customized_content)
    count_query = build_count_query(args.topic_id)
    # 356 is set explicitly for topic 815
    count_response = search(args.index_name, count_query, 356)
    count_rel = count(count_response)
    # print(count_rel)
    response = search(args.index_name, query, args.top_k)
    # reranked by embedding
    if args.vector_name != "":
        result_list = [hit.meta.id for hit in response]
        response = embedding_reranked(result_list,
                                      args.index_name, args.vector_name, args.topic_id, args.query_type, args.top_k)
    # print(count_response.hits.total)
    print("ndcg_score: ", round(generate_ndcg_score(args.topic_id, response), 3))
    res = get_fnfp(response, args.show_fpfn)
    print("false positve:", res[0])
    print("false negative:", res[1])
    print("precision:", round((args.top_k - res[0])/args.top_k, 3))
