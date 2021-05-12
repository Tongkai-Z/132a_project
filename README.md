<!-- HEADER -->
<p align="center">
  <h1 align="center">COSI132 Final Project</h1>
  <p align="center">
    Team member: Tongkai Zhang, Shi Qiu, Bowei Sun
    <br>
    Topic TREC #815: Jason Rezaian released from Iran
    <a herf="https://github.com/Tongkai-Z/132a_project">Github Repo</a>
    <a herf="#">Presentation Slides</a>
  </p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
        <a href="#project-summary">Project Summary</a>
        <ul>
            <li><a href="#intro">Intro</a></li>
            <li><a href="#how-to-run">How to run</a></li>
            <li><a href="#dependencies">Dependencies</a></li>
        </ul>
    </li>
    <li><a href="#">Synonyms Analyzer</a></li>
    <li><a href="#">Query Expansion</a></li>
    <li><a href="#">Bert Model Selection</a></li>
    <li><a href="#">Fine Tune on Bert</a></li>
    <li><a href="#">Contributions</a></li>
  </ol>
</details>

<!-- PROJECT SUMMARY -->
# Project Summary: 
### Intro
* Our project topic: *#815: Jason Rezaian released from Iran*
* Description: Find documents that discuss Washington Post journalist Jason Rezaian release from Iranian prison. 
* Narrative: Relevant documents include those that mention that Washington Post journalist Jason Rezaian had served in an Iranian prison and was released, as well as those that describe efforts from the Washington Post and others to have him released.

We first adopted the metric from HW5 to generate a baseline score on our topics. The baseline socre are shown as below:
<h3>TODO: PUT RESULT TABLE HERE</h3>

To identify the detailed results, we wrote some python script to identify all the false negative and false positive results.
Based on the properties of FP/FN results, we further developed 4 techniques aiming to improve our retrieval results.
1. Apply a synonyms analyzer to generate a new index.
2. Apply Query Expansion
3. Selecting different pre-trained bert models
4. Fine tune on sbert model from HW5 (msmarcos-distilbert-base-v3)
The detailed implementation will be discussed in later sections.

### How to run
<h3>TODO: Revise how to run code</h3>
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
### Dependencies


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

| Query Type              | title | description | narration |
| ------------------------| ----- | ----------- | --------- |
| BM25 + without synonyms | 0.5233| 0.4353      | 0.6389    |
| BM25 + with synonyms    | 0.5026| 0.6348      | 0.5871    |
| fasttext + default      | NA    | NA          | NA        |
| sbert + default         | NA    | NA          | NA        |


#### N-gram analyzer
