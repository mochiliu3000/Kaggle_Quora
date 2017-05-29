## Feature Engineer

#### We used regex to exclude punctuations in the strings and  filtered out the stopwords based on the `nltk.corpus`.

* `word_match_share`
* `tfidf_word_match_share`


#### Added Counting, Distance and more tfidf features(BOW, TFIDF) for ngram = 1, 2, 3
Following ChengLong's idea and code: https://github.com/ChenglongChen/Kaggle_CrowdFlower/tree/master/Code/Feat
```
Count of words
Count of unique words
Ratio of unique words
Count of digits
Count of unique digits
Ratio of unique digits
Count of q1 words in q2
Count of q2 words in q1
Ratio of q1 words in q2
Ratio of q2 words in q1
```

```
jaccard_coef between q1 and q2
dice_dist between q1 and q2
```

```
cosine similarity of q1&q2 tfidf
cosine similarity of q1&q2 BOW
cosine similarity of q1&q2 SVD tfidf
cosine similarity of q1&q2 SVD BOW
```


##  Modeling 

### Simple Models

```
logistic regression -- train AUC scores: 0.804

xgboost -- test AUC scores: 0.872
```
