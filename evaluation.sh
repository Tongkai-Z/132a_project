#!/bin/bash
qt=title

for topic in 815
do
for qt in title description narration
do
echo   
echo $topic $qt
echo bm25_default
python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --top_k 30
echo bm25_c
python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt -u --top_k 30
echo fasttext
python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name ft_vector --top_k 30
echo sbert
python evaluate.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_vector --top_k 30
done
done