# Final Project
**Due**: May 13
```
Team Shi Qiu, Bowen Sun, Tongkai Zhang:
Topic TREC #815
What we did so far:
1. Identify False Negatives and False Positive from baseline results
2. Experiment with different pre-trained models
3. Some False negative results seems neglecting synonyms key terms in query, then we add a synonyms analyzer to solve this
```
Check the slides for more detailed discussion: https://docs.google.com/presentation/d/1YS2NF3w-5RA0q4JEAYsOcV1N4q9_0468drCeWSHY-ns/edit#slide=id.gd766f1b364_0_2
Github Repo: https://github.com/Tongkai-Z/132a_project

## False Negative & False Positive
**False Negative**: not retrieved, relevant
**False Positive**:  retrieved, irrelevant

### Issue1: FN, relevant lvl 2 docs are almost not retrieved
there are 20 level-2 docs in total, but 19 of them are in FN of top_20 retrieved documents

**Solution**: 
	- Synonymous
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

### Suggestions
#### Queries
	- customize the query
		- https://docs.google.com/presentation/d/1eVcXcPLKcIOfszMSfrNH1oh2MsoYPMU1QTTepGkn1SY/edit#slide=id.gd4bdbdc458_0_11
	- query expansion[Google幻灯片 - 用于在线创建和编辑演示文稿，完全免费。](https://docs.google.com/presentation/d/1VaO-38JedXqk2YUYiaY8dkgzBJqIDw0xvWaypY8wAX4/edit#slide=id.gd4a5a6b322_0_0)
	- devide and combine query  vectors
	- Remove redundant words in narrations and descriptions
	- Extract meaningful words, embedding them, Use IDF to filter synonyms
		- [Google幻灯片 - 用于在线创建和编辑演示文稿，完全免费。](https://docs.google.com/presentation/d/19c5Itz2Tn2ZCmPOyZHWKinaVdWd2fVbPXMWAcKl1I4A/edit#slide=id.gd55babf1e4_0_116)
```
Example: “Do college graduates have higher income? Do high-school graduates have higher unemployment?” -> [[college, graduates, high, income], [high-school, graduates, high, unemployment]]
```
#### Embeddings
	- Rank directly with Bert embeddings (no BM25) 
	- creating customized document vectors e.g. doc2vec
		- [Google幻灯片 - 用于在线创建和编辑演示文稿，完全免费。](https://docs.google.com/presentation/d/1eVcXcPLKcIOfszMSfrNH1oh2MsoYPMU1QTTepGkn1SY/edit#slide=id.gd54c5a01c8_0_37)
		- Removing words before training (tf-idf)
		- Training only on relevant documents vs whole corpus
		- Ranking from vectors directly


#### Analyzer
	- N-gram analyzer