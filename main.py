import numpy as np
import pandas as pd
from feature import *
from gen_feat import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df_train = pd.read_csv("train.csv")

#  Applied functions to extract features    
df_train['word_match'] = df_train.apply(word_match_share, axis = 1, raw = True)
df_train['tfdif_word_match'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)

df_train = gen_ngram_data(df_train)
df_train = extract_counting_feat(df_train)
df_train = extract_distance_feat(df_train)
extract_tfidf_feat(df_train)

X = df_train[[
'word_match', 'tfdif_word_match',
       'count_of_q1_unigram', 'count_of_unique_q1_unigram',
       'ratio_of_unique_q1_unigram', 'count_of_q1_bigram',
       'count_of_unique_q1_bigram', 'ratio_of_unique_q1_bigram',
       'count_of_q1_trigram', 'count_of_unique_q1_trigram',
       'ratio_of_unique_q1_trigram', 'count_of_digit_in_q1',
       'ratio_of_digit_in_q1', 'count_of_q2_unigram',
       'count_of_unique_q2_unigram', 'ratio_of_unique_q2_unigram',
       'count_of_q2_bigram', 'count_of_unique_q2_bigram',
       'ratio_of_unique_q2_bigram', 'count_of_q2_trigram',
       'count_of_unique_q2_trigram', 'ratio_of_unique_q2_trigram',
       'count_of_digit_in_q2', 'ratio_of_digit_in_q2',
       'count_of_q1_unigram_in_q2', 'ratio_of_q1_unigram_in_q2',
       'count_of_q2_unigram_in_q1', 'ratio_of_q2_unigram_in_q1',
       'count_of_q1_bigram_in_q2', 'ratio_of_q1_bigram_in_q2',
       'count_of_q2_bigram_in_q1', 'ratio_of_q2_bigram_in_q1',
       'count_of_q1_trigram_in_q2', 'ratio_of_q1_trigram_in_q2',
       'count_of_q2_trigram_in_q1', 'ratio_of_q2_trigram_in_q1',
       'jaccard_coef_of_unigram_between_q1_q2',
       'jaccard_coef_of_bigram_between_q1_q2',
       'jaccard_coef_of_trigram_between_q1_q2',
       'dice_dist_of_unigram_between_q1_q2',
       'dice_dist_of_bigram_between_q1_q2',
       'dice_dist_of_trigram_between_q1_q2', 
       'tfidf_cos_of_q1_q2', 'svd100_tfidf_cos_of_q1_q2',
       'svd150_tfidf_cos_of_q1_q2']]

Y = df_train['is_duplicate']


logi_model = LogisticRegression()
logi_model.fit(X,Y)

logi_y = logi_model.predict_proba(X)[:, 1]
print "Auc scores for Logistic Regression: %s"%(metrics.roc_auc_score(Y, logi_y)) #0.813

print(df_train)