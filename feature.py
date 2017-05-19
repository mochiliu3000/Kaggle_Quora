import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import re, string 

regex = re.compile('[%s]' % re.escape(string.punctuation)) 
stops = set(stopwords.words("english"))

df_train =  pd.read_csv('C:/Users/IBM_ADMIN/Desktop/Machine Learning/quora_nlp/data/train.csv')
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

def get_weights(count, eps = 10, min_count = 2):
	if count < min_count:
		return 0
	else:
		return 1 / float(count**2 + eps)

words = regex.sub("", (" ".join(train_qs))).lower().split()
counts = Counter(words)
weights = {word: get_weights(count) for word, count in counts.items()}

def word_match_share(row):
	q1w = []
	q2w = []
	
	# Remove punctuations
	no_pun_q1 = regex.sub('', str(row['question1']))
	no_pun_q2 = regex.sub('', str(row['question2']))
	
	for word in no_pun_q1.lower().split():
		if word not in stops:
			q1w.append(word)
	
	for word in no_pun_q2.lower().split():
		if word not in stops:
			q2w.append(word)
		
	if len(q1w) * len(q2w) == 0:
		return 0 
	else:
		return len(set(q1w) & set(q2w)) / float(len(set(q1w + q2w)))

def tfidf_word_match_share(row):
	q1w = []
	q2w = []
	
	# Remove punctuations
	no_pun_q1 = regex.sub('', str(row['question1']))
	no_pun_q2 = regex.sub('', str(row['question2']))
	
	for word in no_pun_q1.lower().split():
		if word not in stops:
			q1w.append(word)
	
	for word in no_pun_q2.lower().split():
		if word not in stops:
			q2w.append(word)
		
	if len(q1w) * len(q2w) == 0:
		return 0 
	else:
		share_weights = [weights.get(w,0) for w in list(set(q1w) & set(q2w))]
		union_weights = [weights.get(w,0) for w in list(set(q1w + q2w))]
		return np.sum(share_weights) / float(np.sum(union_weights))
		
		


