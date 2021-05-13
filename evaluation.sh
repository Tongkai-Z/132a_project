#!/bin/bash
qt=title

for topic in 815
do
for qt in title description narration
do
echo   
echo $topic $qt
echo bm25_default
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --top_k 20 
# echo bm25_default+customized_query
# python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --top_k 20 -q
# echo bm25+synonyms_analyzer+customized_query
# python search.py --index_name wapo_docs_50k_synonyms --topic_id $topic --query_type $qt --top_k 20 -u -q
# echo bm25_c
# python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt -u --top_k 20 -fpfn
# echo bm25_c+customized_query
# python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt -u --top_k 20 -fpfn -q
echo fasttext
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name ft_vector --top_k 20 -fpfn
# echo fasttext+customized_query
# python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name ft_vector --top_k 20 -fpfn -q
echo sbert
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_vector --top_k 20
# echo sbert+customized_query
# python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_vector --top_k 20 -q
# echo sbert+synonyms_analyzer+customized_query
# python search.py --index_name wapo_docs_50k_synonyms --topic_id $topic --query_type $qt --vector_name sbert_vector --top_k 20 -u -q

echo sbert_fine_tune
python search.py --index_name wapo_docs_50k --topic_id $topic --query_type $qt --vector_name sbert_fine_tune_vector --top_k 20

done
done
