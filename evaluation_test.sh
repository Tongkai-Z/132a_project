#!/bin/bash
qt=title

for topic in 815
do
for qt in title description narration
do
# echo   
# echo $topic $qt
# echo fasttext
# python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name ft_vector --top_k 20
# echo sbert
# python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_vector --top_k 20

# echo sbert_dpr
# python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_dpr_vector --top_k 20

# echo sbert_dot_product
# python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_dot_product_vector --top_k 20

echo
echo $topic $qt
echo fasttext
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name ft_vector --top_k 20
echo sbert
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_vector --top_k 20

echo sbert_dpr
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_dpr_vector --top_k 20

echo sbert_dot_product
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_dot_product_vector --top_k 20

done
done