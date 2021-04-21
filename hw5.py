from flask import Flask, render_template, request
from evaluate import process_interactive_query
from elasticsearch_dsl.connections import connections

app = Flask(__name__)

result_list = []
query_string = ""
result_dict = {}
search_type = ""


@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    global query_string
    global result_list
    global result_dict  # mapping from id to hit
    global search_type
    query_string = request.form["query"]
    search_type = request.form["search_type"]
    result_list = process_interactive_query(query_string, search_type)
    result_dict = {}
    for hit in result_list:
        result_dict[hit.meta.id] = hit
    # start and end index for first page display
    start = 0
    end = min(len(result_list), 8)
    last = False
    if end == len(result_list):
        last = True
    return render_template("results.html", search_type=search_type, doc_list=result_list, query_text=query_string, start=start, end=end, last=last, page_id=0)


@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    start = page_id * 8
    end = min(len(result_list), start + 8)
    last = False
    if end == len(result_list):
        last = True
    return render_template("results.html", search_type=search_type, doc_list=result_list, query_text=query_string, start=start, end=end, last=last, page_id=page_id)


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    return render_template("doc.html", doc=result_dict[str(doc_id)])


if __name__ == "__main__":
    connections.create_connection(
        hosts=["localhost"], timeout=100, alias="default")
    app.run(debug=True, port=5000)
