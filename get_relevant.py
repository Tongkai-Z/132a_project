import argparse
from utils import parse_wapo_topics
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
from metrics import ndcg
from query import wordnet_query_expansion
from count import build_count_query, search, count, relative_docs
import csv


def get_all_relevant(topic_id, index_name):
    count_query = build_count_query(topic_id)
    count_response = search(index_name, count_query, 356)
    # count_rel = count(count_response)
    # print(len(relative_docs))
    # print(count_rel)
    with open(f"pa5_data/{topic_id}_relevant_docs.csv", mode='w') as csv_file:
        fieldnames = ['label', 'content', 'customed_content']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for hit in count_response:
            if int(hit.annotation[-1]) != 0:
                writer.writerow({'label': '1', 'content': hit.content, 'customed_content': hit.custom_content})
            else:
                writer.writerow({'label': '0', 'content': hit.content, 'customed_content': hit.custom_content})



if __name__ == '__main__':
    connections.create_connection(
        hosts=["localhost"], timeout=100, alias="default")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name", required=True, type=str, help="name of the index"
    )
    parser.add_argument(
        "--topic_id", required=True, type=str, help="topic id"
    )
    args = parser.parse_args()
    get_all_relevant(args.topic_id, args.index_name)

