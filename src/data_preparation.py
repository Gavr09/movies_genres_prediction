import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder 
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, median_absolute_error, roc_auc_score, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
import itertools
import datetime
import tqdm
pd.set_option("display.max_columns", None)

import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def dict_modification(dct):
    '''
    Modification of the initial dictionaries from data frame
    '''
    dct_modif = {}
    for key in dct.keys():
        val = dct[key]
        if type(val) == dict:
            dct_modif[key] = list(val.values())
        else:
            dct_modif[key] = val
    return dct_modif


def read_data(fpath):
    k = 0
    with open(fpath, "r") as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            dct = json.loads(line)
            dct_modif = dict_modification(dct)
            df_tmp = pd.DataFrame([dct_modif])
            if k == 0:
                df = df_tmp.copy()
                k+=1
            else:
                df = pd.concat([df, df_tmp])
    df.index = range(len(df.index))
                
    return df


def clean_text(text):
    '''
    function for text cleaning 
    '''
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def create_res_str(multilabel_binarizer, pred, prob):
    genres_tuple = multilabel_binarizer.inverse_transform(pred)
    res_str = ''
    for i in range(prob.size):
        prob_val = int(prob[i]*100)
        res_str += genres_tuple[0][i] + ' ({}%), '.format(prob_val)
    return res_str[:-2]


def prepare_final_data(data_path, vectorizer_path, model_path, binarazer_path):
    try:
        with open(data_path, "r") as f:
            lines = f.readlines()
        text = r'{}'.format(remove_stopwords(clean_text(lines[0])))
    except:
        print('Incorrect data format! Look at the example')
        return
    
    try:
        tfidf_vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        model = pickle.load(open(model_path, 'rb'))
        multilabel_binarizer = pickle.load(open(binarazer_path, 'rb'))
    except:
        print('Error during unpickling')
        return
    
    X = tfidf_vectorizer.transform(pd.Series(text))
    y_pred_prob = model.predict_proba(X)
    prob = y_pred_prob[y_pred_prob > 0.15]
    y_pred = (y_pred_prob >= 0.15).astype(int)
    return create_res_str(multilabel_binarizer, y_pred, prob)



def create_new_df(N, df):
    '''
    Modificate column with genres in dataframe (df) to leave N most popular genres
    '''
    
    genres_to_leave = all_genres_df.head(N).Genre.values
    
    def genre_cut(init_genre_list):
        new_genre_list = []
        if len(init_genre_list) == 0:
            return np.NaN

        for genre in init_genre_list:
            if genre in genres_to_leave:
                new_genre_list.append(genre)
        if len(new_genre_list) == 0:
            return np.NaN
        else:
            return new_genre_list
    
    df_tmp = df.copy()
    df_tmp['genres_new'] = df_tmp['genres'].apply(lambda x: genre_cut(x))
    df_tmp = df_tmp[df_tmp['genres_new'].isna() == False]
    df_tmp = df_tmp.drop(columns=['genres'])
    df_tmp = df_tmp.rename(columns={'genres_new':'genres'})
    
    return df_tmp
    
def calc_dependency_from_class_num(all_genres_df, df, N_init, N_step):
    '''
    Calculate dependency of model's metrics from the number of the most popular genres, that are used as a target multi-label classes 
    '''
    all_genres_df = all_genres_df.sort_values(['Count'], ascending=False)
    res_df = pd.DataFrame(index=range(100), columns=['N_genres', 'f1_micro', 'precision_micro', 'recall_micro', \
                                                     'f1_macro', 'precision_macro', 'recall_macro', \
                                                     'f1_weighted', 'precision_weighted', 'recall_weighted', \
                                                     'f1_samples', 'precision_samples', 'recall_samples'])
    
    k = 0
    for N in tqdm.tqdm(range(N_init, all_genres_df.shape[0], N_step)):
        new_df = create_new_df(N, df)
        
        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(new_df['genres'])

        # transform target variable
        y = multilabel_binarizer.transform(new_df['genres'])
        
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
        
        # split dataset into training and validation set
        X_train, X_test, y_train, y_test = train_test_split(new_df['clean_plot'], y, test_size=0.2, random_state=9)
        X_train.index = range(len(X_train.index))
        X_test.index = range(len(X_test.index))
        
        # create TF-IDF features
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        lr = LogisticRegression()
        clf_lr = OneVsRestClassifier(lr)
        
        clf_lr.fit(X_train_tfidf, y_train)
        y_pred_prob = clf_lr.predict_proba(X_test_tfidf)
        
        t_list = [i for i in np.arange(0, 0.7, 0.01)]
        f1_score_list = []
        for t in t_list:
            y_pred_new = (y_pred_prob >= t).astype(int)
            f1_score_list.append(f1_score(y_test, y_pred_new, average="weighted"))
        
        f1_max = max(f1_score_list)
        idx = f1_score_list.index(f1_max)
        t_opt = t_list[idx]

        y_pred_new = (y_pred_prob >= t_opt).astype(int)
        
        f1_micro = f1_score(y_test, y_pred_new, average="micro")
        precision_micro = precision_score(y_test, y_pred_new, average="micro")
        recall_micro = recall_score(y_test, y_pred_new, average="micro")
        
        f1_macro = f1_score(y_test, y_pred_new, average="macro")
        precision_macro = precision_score(y_test, y_pred_new, average="macro")
        recall_macro = recall_score(y_test, y_pred_new, average="macro")
        
        f1_weighted = f1_score(y_test, y_pred_new, average="weighted")
        precision_weighted = precision_score(y_test, y_pred_new, average="weighted")
        recall_weighted = recall_score(y_test, y_pred_new, average="weighted")
        
        f1_samples = f1_score(y_test, y_pred_new, average="samples")
        precision_samples = precision_score(y_test, y_pred_new, average="samples")
        recall_samples = recall_score(y_test, y_pred_new, average="samples")
        
        
        res_df.loc[k, 'N_genres'] = N
        res_df.loc[k, 'f1_micro'] = f1_micro
        res_df.loc[k, 'precision_micro'] = precision_micro
        res_df.loc[k, 'recall_micro'] = recall_micro
        
        res_df.loc[k, 'f1_macro'] = f1_macro
        res_df.loc[k, 'precision_macro'] = precision_macro
        res_df.loc[k, 'recall_macro'] = recall_macro
        
        res_df.loc[k, 'f1_weighted'] = f1_weighted
        res_df.loc[k, 'precision_weighted'] = precision_weighted
        res_df.loc[k, 'recall_weighted'] = recall_weighted
        
        res_df.loc[k, 'f1_samples'] = f1_samples
        res_df.loc[k, 'precision_samples'] = precision_samples
        res_df.loc[k, 'recall_samples'] = recall_samples
        
        k+=1
        
        res_df.to_csv('model_performance_results.csv', index=False)
        
    res_df = res_df.dropna()
    res_df.to_csv('model_performance_results.csv', index=False)
       
    return res_df