import numpy as np
import pandas as pd
import cPickle
from feature import *
from gen_feat import *
from sklearn.linear_model import LogisticRegression
from sklearn import metirc
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import xgboost as xgb



def logit_fit(X, Y):
	
	#  Logistic regression
	logi_fit  = LogisticRegression()
	logi_fit.fit(X,Y)
	logi_pred = logi_fit.predict_proba(X = X)[:,1]
	print 'Logistics Regression have auc: %s'%(metrics.roc_auc_score(Y, logi_pred))  # 0.804
	
	return logi_fit


def xgb_fit(X, Y):
	'''
	Since I have already trained the xgb models and dumped the objects to local folder,
	I comment all the codes for training. And Load the object into script directly to avoid 
	unnecessary repeatted works.
	'''  

# 
# 	gbm = xgb.XGBClassifier(n_estimators = 1000, 
# 							subsample=0.8, 
# 							colsample_bytree=0.8, 
# 							objective= 'binary:logistic',
# 							nthread = -1)
# 	gbm_params = {
# 		'learning_rate': [0.05, 0.1],
# 		'max_depth': range(3, 10, 2),
# 		'min_child_weight':[1, 3, 5],
# 		'reg_alpha':[1e-5, 1e-2, 0.1]  
# 	}
# 	cv = StratifiedKFold(Y)
# 	grid = GridSearchCV(gbm, gbm_params,scoring='roc_auc',cv=cv,verbose=10,n_jobs=-1)
# 	grid.fit(X, Y)
# 	
# 
# 	 Dumped XGB models into models folder
# 	para_name = ""
# 	for k, v in grid.best_params_.items():
# 		para_name += "_%s_%s"%(k,v)
# 
# 	with open("models/AL_xgb%s.pkl"%(para_name), "wb") as f:
# 		cPickle.dump(grid, f, -1)
	
	grid = cPickle.load(open("models/AL_xgb_learning_rate_0.1_reg_alpha_0.1_min_child_weight_1_max_depth_9.pkl", "rb"))
	
	
	print (grid.best_params_, grid.best_score_) 
	
	'''
	({'learning_rate': 0.1, 'reg_alpha': 0.1, 'min_child_weight': 1, 'max_depth': 9}, 0.8720490539202848)
	'''
	

	return grid.best_estimator_  # return best estimator for xgb







if __name__ == "__main__":

	#  Loaded data
	df_train = pd.read_csv("train.csv")

	#  Applied functions to extract features	
	df_train['word_match'] = df_train.apply(word_match_share, axis = 1, raw = True)
	df_train['tfdif_word_match'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)

	df_train = gen_ngram_data(df_train)
	df_train = extract_counting_feat(df_train)
	df_train = extract_distance_feat(df_train)
	#extract_tfidf_feat(df_train)

	X = df_train[[ 'word_match', 'tfdif_word_match',
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
		   'dice_dist_of_trigram_between_q1_q2']]

	Y = df_train['is_duplicate']
	
	
	
	# Logistic Regression
	
	logi_model = logit_fit(X, Y)
	
	# XGB
	
	xgb_model = xgb_fit(X, Y)







