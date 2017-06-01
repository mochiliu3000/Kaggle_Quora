from nlp_utils import stopwords, english_stemmer, stem_tokens, \
    compute_dist, try_divide, cat_text, getTFV, getBOW, cosine_sim
from sklearn.decomposition import TruncatedSVD
from param_config import config
import cPickle
import ngram
import numpy as np
import pandas as pd
import sys
from scipy.sparse import vstack

reload(sys)
sys.setdefaultencoding('utf8')
svd_n_components = [100, 50]


# Tokenize and stem the data
def preprocess_data(line, exclude_stopword=True):
    ## tokenize
    tokens = [x.lower() for x in str(line).split()]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed

# Generate ngram data (n = 1, 2, 3)
def gen_ngram_data(df):
    ## unigram
    print("generate unigram")
    df["q1_unigram"] = list(df.apply(lambda x: preprocess_data(x["question1"]), axis=1))
    df["q2_unigram"] = list(df.apply(lambda x: preprocess_data(x["question2"]), axis=1))
    ## bigram
    print("generate bigram")
    join_str = "_"
    df["q1_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["q1_unigram"], join_str), axis=1))
    df["q2_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["q2_unigram"], join_str), axis=1))
    ## trigram
    print("generate trigram")
    join_str = "_"
    df["q1_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["q1_bigram"], join_str), axis=1))
    df["q2_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["q2_bigram"], join_str), axis=1))
    return df

# Extract counting features based on ngram generated
def extract_counting_feat(df):
    feat_names = ["q1", "q2"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    ################################
    ## word count and digit count ##
    ################################
    print("generate basic counting features...")
    for feat_name in feat_names:
        for gram in grams:
            ## word count
            df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
            df["count_of_unique_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
            df["ratio_of_unique_%s_%s" % (feat_name, gram)] = map(try_divide, df["count_of_unique_%s_%s" % (feat_name, gram)], df["count_of_%s_%s" % (feat_name, gram)])

        ## digit count
        df["count_of_digit_in_%s" % feat_name] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
        df["ratio_of_digit_in_%s" % feat_name] = map(try_divide, df["count_of_digit_in_%s" % feat_name], df["count_of_%s_unigram" % feat_name])

    #########################
    ## interact word count ##
    #########################
    print("generate interact counting features...")
    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                if target_name != obs_name:
                    # shared words
                    df["count_of_%s_%s_in_%s" % (obs_name, gram, target_name)] = list(df.apply(
                        lambda x: sum([1. for w in x[obs_name + "_" + gram] if w in set(x[target_name + "_" + gram])]), axis=1))
                    df["ratio_of_%s_%s_in_%s" % (obs_name, gram, target_name)] = map(try_divide,
                        df["count_of_%s_%s_in_%s" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
    return df

# Extract distance features based on ngram generated
def extract_distance_feat(df):
    ## jaccard coef/dice dist of n-gram
    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["q1", "q2"]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names) - 1):
                for j in range(i + 1, len(feat_names)):
                    target_name = feat_names[i]
                    obs_name = feat_names[j]
                    df["%s_of_%s_between_%s_%s" % (dist, gram, target_name, obs_name)] = \
                        list(df.apply(
                            lambda x: compute_dist(x[target_name + "_" + gram], x[obs_name + "_" + gram], dist), axis=1))
    return df

# Extract tfidf features based on ngram generated
def extract_tfidf_feat(df):
    df["all_text"] = list(df.apply(cat_text, axis=1))
    vec_types = ["tfidf", "bow"]
    feat_names = ["question1", "question2"]
    for vec_type in vec_types:
        if vec_type == "tfidf":
            vec = getTFV(ngram_range=(1,3))
        elif vec_type == "bow":
            vec = getBOW(ngram_range=(1,3))

        # get common vocabulary
        vec.fit(df["all_text"])
        vocabulary = vec.vocabulary_
        print("generate ngram %s feat for %s" % (vec_type, feat_names[0]))
        if vec_type == "tfidf":
            vec = getTFV(ngram_range=(1, 3), vocabulary=vocabulary)
        elif vec_type == "bow":
            vec = getBOW(ngram_range=(1, 3), vocabulary=vocabulary)

        # fit common vocabulary on each specific question
        q1_vec = vec.fit_transform(df[feat_names[0]])
#        with open("%s/train.%s.%s.pkl" % (config.processed_data_path, feat_names[0], vec_type), "wb") as f:
#            cPickle.dump(q1_vec, f, -1)
        q2_vec = vec.fit_transform(df[feat_names[1]])
#        with open("%s/train.%s.%s.pkl" % (config.processed_data_path, feat_names[1], vec_type), "wb") as f:
#            cPickle.dump(q2_vec, f, -1)
        print("q1_vec has shape: %s, while q2_vec has shape: %s" % (q1_vec.shape, q2_vec.shape))

        # calculate Cos distance of these 2 vecs
        print("generate common %s cosine sim feat for q1 and q2" % vec_type)
        df["%s_cos_of_q1_q2" % vec_type] = np.asarray(map(cosine_sim, q1_vec, q2_vec))[:, np.newaxis]

        # calculate SVD Cos distance of these 2 vecs
        print("generate svd %s cosine sim feat for q1 and q2" % vec_type)
        # vertically stack q1 and q2
        q1_q2_vec = vstack([q1_vec, q2_vec])
        for n_components in svd_n_components:
            svd = TruncatedSVD(n_components=n_components, n_iter=15)
            svd.fit(q1_q2_vec)
            q1_svd_vec = svd.transform(q1_vec)
            q2_svd_vec = svd.transform(q2_vec)
            print("q1_svd_vec has shape: %s, while q2_svd_vec has shape: %s" % (q1_svd_vec.shape, q2_svd_vec.shape))
            df["svd%s_%s_cos_of_q1_q2" % (n_components, vec_type)] = np.asarray(map(cosine_sim, q1_svd_vec, q2_svd_vec))[:, np.newaxis]

    return df

'''
if __name__ == "__main__":
    dfTrain_path = "%s/train.csv" % config.data_path
    dfTest_path = "%s/test.csv" % config.data_path
    dfTrain = pd.read_csv(dfTrain_path)
    dfTest = pd.read_csv(dfTest_path)
    print(dfTrain.head())


    # Generate ngram = 1, 2, 3
    dfTrain = gen_ngram_data(dfTrain)
    # dfTest = gen_ngram_data(dfTest)

    # Extract counting features
    dfTrain_count = extract_counting_feat(dfTrain)
    # dfTest_count = extract_counting_feat(dfTest)
    with open("%s/train.gen.count.pkl" % config.processed_data_path, "wb") as f:
        cPickle.dump(dfTrain_count, f, -1)
    # with open("%s/test.gen.count.pkl" % config.processed_data_path, "wb") as f:
    #     cPickle.dump(dfTest_count, f, -1)
    print("Dumped counting features to df...")

    print(dfTrain_count.head())

    # Extract distance features
    dfTrain_dist = extract_distance_feat(dfTrain_count)
    with open("%s/train.gen.dist.pkl" % config.processed_data_path, "wb") as f:
        cPickle.dump(dfTrain_dist, f, -1)
    # with open("%s/test.gen.dist.pkl" % config.processed_data_path, "wb") as f:
    #     cPickle.dump(dfTest_dist, f, -1)
    print("Dumped distance features to df...")

    # Extract tfidf features
    dfTrain_dist_tfidf = extract_tfidf_feat(dfTrain_dist)
    with open("%s/train.gen.tfidf.pkl" % config.processed_data_path, "wb") as f:
        cPickle.dump(dfTrain_dist_tfidf, f, -1)
    print(dfTrain_dist_tfidf.head())
'''
