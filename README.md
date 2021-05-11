# Final Project

**Due**: May 13
Team Shi Qiu, Bowen Sun, Tongkai Zhang:
Topic TREC #815
What we did so far:

**Author**: Tongkai Zhang, Shi Qiu, Bowei Sun

**Date**: May 4, 2021

1. Identify False Negatives and False Positive from baseline results
2. Experiment with different pre-trained models
3. Some False negative results seems neglecting synonyms key terms in query, then we add a synonyms analyzer to solve this

Check the slides for more detailed discussion: https://docs.google.com/presentation/d/1YS2NF3w-5RA0q4JEAYsOcV1N4q9_0468drCeWSHY-ns/edit#slide=id.gd766f1b364_0_2

Github Repo: https://github.com/Tongkai-Z/132a_project

## How to run

```
conda activate cosi132a
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3
python count.py --index_name wapo_docs_50k --topic_id 815 --query_type narration --vector_name sbert_vector --top_k 20

# Run synonyms analyzer
1. Build new index
python load_es_index.py --index_name wapo_docs_50k_synonyms --wapo_path pa5_data/subset_wapo_50k_sbert_ft_filtered.jl
2. Run evaluation based on new index
python count.py --index_name wapo_docs_50k_synonyms --topic_id 815 --query_type description --vector_name sbert_vector --top_k 20 -u

# script
sh evaluation.sh



```

## False Negative & False Positive

**False Negative**: not retrieved, relevant
**False Positive**: retrieved, irrelevant

### Issue1: FN, relevant lvl 2 docs are almost not retrieved

there are 20 level-2 docs in total, but 19 of them are in FN of top_20 retrieved documents

**Solution**: - Synonymous

```
To improve FP and FN results:
	1.	FP: The documents contains more keywords,  but not highly related to the description retrieved. (Trial of Post reporter detained in Iran may be nearing end) Mentions about trails, but about serving and releasing from the prision. ( State Department urges Iran to release Washington Post correspondent) mentions the actions from US government, but also not including the effort to release the reporter neigher him serving in prison.

	2.	FN: Some documents that are highly related, not but containing the keywords are not selected. ( State Department urges Iran to release Washington Post correspondent) This document is about the effort from US Government, but the word “urges”, “free” is not mentioned in description, thus not selected.


We suspect the reason is that Bert is not as effective as expected. Also there are some terms in FN documents, such as “urges”, “to free” are not considered as relative terms, The possible solution is to apply some synonymous in the analyzer. Also fine tune bert with highly relevant documents would probably improve the effectiveness of Bert.
```

## Possible Approach

- [ ] Try different pre-trained bert. (Bert on larger corpus, different selected documents) Need to find resources online and integrate into Elasticsearch
- [ ] Train bert with our selected, relevant documents. Need to implement our own code to build and train bert. How to intergrate the tuned bert to elasticsearch.
- [ ] Test out different keyword searches. 1)including more relevant keywords 2) conjunction of two piece of queries.
- [ ] Adopt the baseline metric from other team, such as MCDA, TREC#803

## Progress

1. pre-trained model
2. Synonymous
3. Retrieve based on fasttext/sbert embedding's only, Not using BM25 as baseline

### Suggestions

#### Queries

- customize the query
  - https://docs.google.com/presentation/d/1eVcXcPLKcIOfszMSfrNH1oh2MsoYPMU1QTTepGkn1SY/edit#slide=id.gd4bdbdc458_0_11
- query expansion[Google 幻灯片 - 用于在线创建和编辑演示文稿，完全免费。](https://docs.google.com/presentation/d/1VaO-38JedXqk2YUYiaY8dkgzBJqIDw0xvWaypY8wAX4/edit#slide=id.gd4a5a6b322_0_0)
- devide and combine query vectors
- Remove redundant words in narrations and descriptions
- Extract meaningful words, embedding them, Use IDF to filter synonyms
  - [Google 幻灯片 - 用于在线创建和编辑演示文稿，完全免费。](https://docs.google.com/presentation/d/19c5Itz2Tn2ZCmPOyZHWKinaVdWd2fVbPXMWAcKl1I4A/edit#slide=id.gd55babf1e4_0_116)

```
Example: “Do college graduates have higher income? Do high-school graduates have higher unemployment?” -> [[college, graduates, high, income], [high-school, graduates, high, unemployment]]
```

#### Embeddings

- Rank directly with Bert embeddings (no BM25)
- creating customized document vectors e.g. doc2vec
  - [Google 幻灯片 - 用于在线创建和编辑演示文稿，完全免费。](https://docs.google.com/presentation/d/1eVcXcPLKcIOfszMSfrNH1oh2MsoYPMU1QTTepGkn1SY/edit#slide=id.gd54c5a01c8_0_37)
  - Removing words before training (tf-idf)
  - Training only on relevant documents vs whole corpus
  - Ranking from vectors directly

#### Add synonyms analyzer

Based on HW5, we added a synonyms analyzer. The synonyms mapping is the unretrieved terms in the FN results list, such as
release is synonyms of "effort， urges, to free, released, nearing end", other parties is synonyms of
"Washington Post, Jeff Bezos, National Press Club, U.N. human rights experts".

We generated a new index called _wapo_docs_50k_synonyms_ to test out the effect.

To run the new analyzer, first generate a new index with customized analyzer. Then run evaluation metrics on the new index.

| Query Type              | title  | description | narration |
| ----------------------- | ------ | ----------- | --------- |
| BM25 + without synonyms | 0.5233 | 0.4353      | 0.6389    |
| BM25 + with synonyms    | 0.5026 | 0.6348      | 0.5871    |
| fasttext + default      | NA     | NA          | NA        |
| sbert + default         | NA     | NA          | NA        |

#### N-gram analyzer

## Query Expansion

As it is analyzed in our baseline searches, the tier-2 relevant document is rarely retrieved by all methods. Based on the content of the tier-2 documents, one observation is that these document contains the exact information we need, but present using other expressions. Thus, query expansion is to broadens the query by introducing additional tokens or phrases. In our project, we use the automatic query expansion model, so that this mechanism can be applied to any queries under any topic.

### Wordnet Synonyms Expansion

Query is expanded based on the synonyms of each term. Basically add the synonyms to the near position of each term.

Different threshold is experimented for the query. Threshold x means for each term there will be at most x synonyms added.

**Example**:
Jason Rezaian released from Iran

Threshold 2:
Jason let_go_of let_go release Iran Islamic_Republic_of_Iran Persia

Threshold 3:
Jason let_go_of let_go release relinquish Iran Islamic_Republic_of_Iran Persia

**Result**

Threshold 3: 

(ndcg@20score/precision)

| Query Type                     | Title     | Narration  | Description |
| ------------------------------ | --------- | ---------- | ----------- |
| bm25                           | 0.523/0.2 | 0.435/0.15 | 0.64/0.25   |
| bm25 + qe(query expansion)     | 0.787/0.4 | 0.59/0.15  | 0.613/0.25  |
| bm25 + synonyms_analyzer + qe  | 0.584/0.3 | 0.59/0.2   | 0.444/0.15  |
| sbert                          | 0.627/0.2 | 0.878/0.15 | 0.612/0.25  |
| sbert + qe                     | 0.784/0.4 | 0.342/0.15 | 0.428/0.25  |
| sbert + synonyms_analyzer + qe | 0.803/0.3 | 0.375/0.2  | 0.364/0.15  |

