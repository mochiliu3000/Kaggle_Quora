import numpy as np
import pandas as pd
from feature import *


#  Loaded data
df_train = pd.read_csv("C:/Users/IBM_ADMIN/Desktop/Machine Learning/quora_nlp/data/train.csv")

#  Applied functions to extract features	
df_train['word_match'] = df_train.apply(word_match_share, axis = 1, raw = True)
df_train['tfdif_word_match'] = df_train.apply(tfidf_word_match_share, axis = 1, raw = True)
print(df_train)