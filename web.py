from flask import Flask, render_template, request
from search import process_interactive_query
from elasticsearch_dsl.connections import connections

'''
This module integrates the search engine backend with a simple front end, mainly for improve research efficiency

@author Tongkai Zhang
'''

app = Flask(__name__)

result_list = []
topic_id = None
result_dict = {}
search_type = ""
query_type = ""
fn = []
fp = []
precision = 0


@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    global topic_id
    global result_list
    global result_dict  # mapping from id to hit
    global search_type
    global score
    global query_string
    global fn
    global fp
    global precision
    topic_id = request.form["topic_id"]
    query_expansion = request.form["query_expansion"]
    analyzer = request.form["analyzer"]
    query_type = request.form["query_type"]
    embedding_type = request.form["embedding_type"]
    query_string = request.form["query_string"]
    search_type = 'Query_expansion: %s, Analyzer: %s, Query_type: %s, Embedding_type: %s' % (
        query_expansion, analyzer, query_type, embedding_type)
    print(search_type)
    result = process_interactive_query(
        topic_id, query_expansion, analyzer, query_type, embedding_type, query_string)
    result_list = result[0]
    score = result[1]
    query_string = result[2]
    fn = result[3][0]
    fp = result[3][1]
    precision = round((20 - len(fp))/20, 3)
    result_dict = {}
    for hit in result_list:
        result_dict[hit.meta.id] = hit
    for hit in fn:
        result_dict[hit.meta.id] = hit
    # start and end index for first page display
    start = 0
    end = min(len(result_list), 8)
    last = False
    if end == len(result_list):
        last = True
    return render_template("results.html", precision=precision, score=score, search_type=search_type, doc_list=result_list, query_text=query_string, start=start, end=end, last=last, page_id=0)


@ app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    start = page_id * 8
    end = min(len(result_list), start + 8)
    last = False
    if end == len(result_list):
        last = True
    return render_template("results.html", precision=precision, score=score, search_type=search_type, doc_list=result_list, query_text=topic_id, start=start, end=end, last=last, page_id=page_id)


@ app.route("/fn")
def false_negative():
    return render_template("fn.html", fn=fn)


@ app.route("/fp")
def false_positive():
    return render_template("fp.html", fp=fp)


@ app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    return render_template("doc.html", doc=result_dict[str(doc_id)])


if __name__ == "__main__":
    connections.create_connection(
        hosts=["localhost"], timeout=100, alias="default")
    app.run(debug=True, port=5000)
