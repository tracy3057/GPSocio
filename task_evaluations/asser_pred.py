import torch
import pandas as pd
import json
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import itertools
from sklearn.metrics import ndcg_score
import statistics
import torch.nn as nn

def main():

    raw_embedding = torch.load("target_data/preprocess/item_embeddings_target_data")
    raw_seq_embedding = json.load(open("target_data/preprocess/asser_embeddings_target_data.json"))
    embedding = torch.load("target_data/preprocess/propagated_item_embedding_asser_pred")
    seq_embedding = json.load(open("target_data/preprocess/propagated_asser_embedding_asser_pred.json"))
    meta = json.load(open("target_data/meta_data.json"))
    user_map = json.load(open("target_data/smap.json"))
    assertion_map = json.load(open("target_data/umap.json"))
    test = json.load(open("target_data/test.json"))

    data = pd.read_pickle("target_data/selected_df_sequence_test.pkl")
    assertion_author_df = data[['assertion_id','author_id']].explode('author_id').reset_index(drop=True).groupby(['author_id'])['assertion_id'].apply(list).reset_index()
    assertion_author_df['test_asser'] = assertion_author_df['assertion_id'].apply(lambda x: x[-1]).astype(str)
    test_asser_mapped = []
    for idx, row in assertion_author_df.iterrows():
        test_asser_mapped.append(assertion_map[row['test_asser']])
    assertion_author_df['test_asser'] = test_asser_mapped
    test_asser_list = list(set(assertion_author_df['test_asser']))
    new_asser_id_list = []
    for idx, row in assertion_author_df.iterrows():
        new_asser_id_list.append([i for i in row['assertion_id'] if i not in test_asser_list])
    
    assertion_author_df['assertion_id'] = new_asser_id_list
    assertion_author_df = assertion_author_df.loc[assertion_author_df['assertion_id'].map(len)>2]



    def seq_future_pred(embedding,seq_embedding,meta,user_map,assertion_map,test,assertion_author_df):
        user_emb = embedding
        asser_emb = seq_embedding
        meta_data = meta
        cos = nn.CosineSimilarity()
        user_emb_df = pd.DataFrame()
        user_raw_id_list = list(user_map.keys())
        user_id_list = list(user_map.values())

        user_emb_values = user_emb
        user_emb_df['user_id'] = user_id_list
        user_emb_df['user_raw_id'] = user_raw_id_list
        user_emb_df['user_emb_values'] = user_emb_values.cpu().tolist()
        new_user_list = list(assertion_author_df.author_id.values)
        user_emb_df = user_emb_df.loc[user_emb_df['user_raw_id'].isin(new_user_list)]
        user_emb_df = user_emb_df.reset_index()
        user_emb_dict = dict(zip(user_emb_df['user_raw_id'], user_emb_df['user_emb_values']))
        asser_emb_array = []
        assertion_author_df['test_asser'] = assertion_author_df['test_asser'].astype(str)
        test_asser_list = list(set(list(assertion_author_df.test_asser.values)))
        asser_id = []
        for test_asser in test_asser_list:
            try:
                asser_emb_array.append(asser_emb[test_asser])
                asser_id.append(test_asser)
            except:
                continue
        asser_emb_df = pd.DataFrame()
        asser_emb_values = list(asser_emb_array)
        asser_emb_df['asser_id'] = asser_id
        asser_emb_df['asser_emb_values'] = asser_emb_values
        asser_emb_df = asser_emb_df.reset_index()
        asser_emb_df['asser_id']  = asser_emb_df['asser_id'].astype(str)


        user_emb_df_list = []
        for idx, row in assertion_author_df.iterrows():

            cur_user_emb = list(user_emb_dict[row['author_id']])
            user_emb_df_list.append(cur_user_emb)
        assertion_author_df['user_seq_emb'] = user_emb_df_list
        assertion_author_df['test_asser'] = assertion_author_df['test_asser'].astype(str)


        N=len(asser_emb_array)
        pred_assers_list = []
        y_true = []
        y_pred = []
        assertion_author_df = assertion_author_df.loc[assertion_author_df['user_seq_emb'].map(len)>0]
        asser_idx_order = test_asser_list
        for idx, row in assertion_author_df.iterrows():
            seq_emb = np.asarray([row['user_seq_emb']])
            res = cos(torch.tensor(seq_emb), torch.tensor(asser_emb_array)).tolist()
            l_idx = sorted(range(len(res)), key=lambda k: res[k])[::-1]
            pred_assers = list(asser_emb_df.loc[l_idx,:].asser_id.values)
            pred_assers = [int(i) for i in pred_assers]

            cur_true = [0 for i in range(N)]
            try:
                cur_true[asser_id.index(row['test_asser'])] = 1
            except:
                pass
            pred_assers_list.append(pred_assers)
            y_true.append(cur_true)
            y_pred.append(res)

        assertion_author_df['pred_assers'] = pred_assers_list
        assertion_author_df['y_true'] = y_true
        assertion_author_df['y_pred'] = y_pred
     
        assertion_author_df = assertion_author_df.loc[~assertion_author_df['y_true'].apply(lambda x: all(v == 0 for v in y_true))]
        total_cnt_user = 0
        recall_5_list_user = []
        recall_10_list_user = []
        recall_20_list_user = []
        recall_50_list_user = []
        for idx, row in assertion_author_df.iterrows():

            total_cnt_user += 1
            cur_test_asser = set([int(row['test_asser'])])
            recall_5_list_user.append(len(set(row['pred_assers'][0:5]).intersection(cur_test_asser)))
            recall_10_list_user.append(len(set(row['pred_assers'][0:10]).intersection(cur_test_asser)))
            recall_20_list_user.append(len(set(row['pred_assers'][0:20]).intersection(cur_test_asser)))
            recall_50_list_user.append(len(set(row['pred_assers'][0:50]).intersection(cur_test_asser)))
        avg_recall_5_user = statistics.mean(recall_5_list_user)
        avg_recall_10_user = statistics.mean(recall_10_list_user)
        avg_recall_20_user = statistics.mean(recall_20_list_user)
        avg_recall_50_user = statistics.mean(recall_50_list_user)
        y_true_array = np.asarray(list(assertion_author_df.y_true.values))
        y_pred_array = np.asarray(list(assertion_author_df.y_pred.values))
        ndcg_5 = ndcg_score(y_true_array, y_pred_array, k=5)
        ndcg_10 = ndcg_score(y_true_array, y_pred_array, k=10)
        ndcg_20 = ndcg_score(y_true_array, y_pred_array, k=20)
        ndcg_50 = ndcg_score(y_true_array, y_pred_array, k=50)
        print('recall',avg_recall_5_user,avg_recall_10_user,avg_recall_20_user,avg_recall_50_user)
        print('ndcg',ndcg_5,ndcg_10,ndcg_20,ndcg_50)
    

    print("after propogation")
    seq_future_pred(embedding,seq_embedding,meta,user_map,assertion_map,test,assertion_author_df)


main()