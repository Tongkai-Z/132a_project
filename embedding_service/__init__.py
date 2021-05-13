# set the port number for an embedding server
# we may want to be consistent about the port number for different servers to avoid conflicts.
PORT_EMBEDDING_MAPPING = {8080: "sbert", 8081: "fasttext", 8082: "sbert_dot_product", 8083: "sbert_dpr", 8084: "sbert_dot_product"}
INV_PORT_EMBEDDING_MAPPING = {"sbert": 8080, "fasttext": 8081, "sbert_dot_product": 8082, "sbert_dpr": 8083, "sbert_dot_product": 8084}
