import numpy as np
import pandas as pd
import cPickle
from feature import *
from gen_feat import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from param_config import config



def logit_fit(X, Y):

    #  Logistic regression
    logi_fit  = LogisticRegression()
    logi_fit.fit(X,Y)
    logi_pred = logi_fit.predict_proba(X = X)[:,1]
    print 'Logistics Regression have auc: %s'%(metrics.roc_auc_score(Y, logi_pred))  # 0.804

    return logi_fit


def xgb_fit(X, Y, X_test):
    '''
    Since I have already trained the xgb models and dumped the objects to local folder,
    I comment all the codes for training. And Load the object into script directly to avoid
    unnecessary repeatted works.
    '''


    gbm = xgb.XGBClassifier(n_estimators = 1500,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective= 'binary:logistic',
                            nthread = -1)
    gbm_params = {
        'learning_rate': [0.05, 0.07, 0.1, 0.3],
        'max_depth': range(3, 10, 1),
        'min_child_weight': [1, 2, 3, 4, 5],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 5]
    }
    cv = StratifiedKFold(Y)
    grid = GridSearchCV(gbm, gbm_params,scoring='roc_auc',cv=cv,verbose=10,n_jobs=-1)
    grid.fit(X, Y)

    best_model = xgb.XGBRegressor(**grid.best_params_).fit(X, Y)
    pred = best_model.predict(X_test)

    # Dumped XGB models into models folder
    para_name = ""
    for k, v in grid.best_params_.items():
        para_name += "_%s_%s"%(k,v)

    with open("models/xgb_AL_xgb%s.pkl"%(para_name), "wb") as f:
        cPickle.dump(grid, f, -1)

    # grid = cPickle.load(open("models/AL_xgb_learning_rate_0.1_reg_alpha_0.1_min_child_weight_1_max_depth_9.pkl", "rb"))

    submission = pd.read_csv("%s/sample_submission.csv" % config.data_path)
    submission['is_duplicate'] = pred

    submission.to_csv("%s/xgb_pred.csv" % config.data_path, index=False)
    print (grid.best_params_, grid.best_score_)

    '''
    ({'learning_rate': 0.1, 'reg_alpha': 0.1, 'min_child_weight': 1, 'max_depth': 9}, 0.8720490539202848)
    '''


    return grid.best_estimator_  # return best estimator for xgb


if __name__ == "__main__":

    #  Loaded data
    df_train = pd.read_csv("%s/train.csv" % config.data_path)
    df_test = pd.read_csv("%s/test.csv"  % config.data_path)

    #  Applied functions to extract features
    df_train['word_match'] = df_train.apply(word_match_share, axis = 1, raw = True)
    df_train['tfdif_word_match'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)
    df_test['word_match'] = df_test.apply(word_match_share, axis = 1, raw = True)
    df_test['tfdif_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

    df_train = gen_ngram_data(df_train)
    df_train = extract_counting_feat(df_train)
    df_train = extract_distance_feat(df_train)
    df_train = extract_tfidf_feat(df_train)
    df_test = gen_ngram_data(df_test)
    df_test = extract_counting_feat(df_test)
    df_test = extract_distance_feat(df_test)
    df_test = extract_tfidf_feat(df_test)

    col_names = df_train.columns.values

    feat = [ 'word_match', 'tfdif_word_match',
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
         'dice_dist_of_trigram_between_q1_q2', 'tfidf_cos_of_q1_q2', 
         'bow_cos_of_q1_q2', 'svd100_tfidf_cos_of_q1_q2', 
         'svd100_tfidf_cos_of_q1_q2', 'svd50_bow_cos_of_q1_q2', 
         'svd50_bow_cos_of_q1_q2']

    X_train = df_train[feat]
    Y_train = df_train['is_duplicate']
    X_test = df_test[feat]

    # Logistic Regression
    # logi_model = logit_fit(X_train, Y_train)

    # XGB
    xgb_model = xgb_fit(X_train, Y_train, X_test)
