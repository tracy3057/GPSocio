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
    embedding = torch.load("target_data/preprocess/propagated_item_embedding")
    seq_embedding = json.load(open("target_data/preprocess/propagated_asser_embedding.json"))
    meta = json.load(open("target_data/meta_data.json"))
    user_map = json.load(open("target_data/smap.json"))
    assertion_map = json.load(open("target_data/umap.json"))
    test = json.load(open("target_data/test.json"))

    def seq_future_pred(embedding,seq_embedding,meta,user_map,assertion_map,test):
        user_emb = embedding
        asser_emb = seq_embedding
        meta_data = meta
        cos = nn.CosineSimilarity()
        user_emb_df = pd.DataFrame()
        author_id_list = list(meta.keys())
        user_id_list = list(user_map.values())

        # user_emb_values = user_emb.tolist()
        user_emb_values = user_emb
        user_emb_df['user_id'] = user_id_list
        user_emb_df['user_emb_values'] = user_emb_values.cpu().tolist()
        user_emb_df = user_emb_df.reset_index()
        # print(user_emb_df)

        asser_emb_df = pd.DataFrame()
        asser_id = list(asser_emb.keys())

        asser_emb_values = list(asser_emb.values())
        asser_emb_df['asser_id'] = asser_id
        asser_emb_df['asser_emb_values'] = asser_emb_values
        asser_emb_df['next_user'] = list(test.values())
        asser_emb_df = asser_emb_df.reset_index()

        user_emb_array = list(user_emb_df.user_emb_values.values)

        N=len(user_emb_array)
        pred_assers_list = []
        y_true = []
        y_pred = []
        for idx, row in asser_emb_df.iterrows():
            seq_emb = np.asarray([row['asser_emb_values']])
            res = cos(torch.tensor(seq_emb), torch.tensor(user_emb_array)).tolist()
            l_idx = sorted(range(len(res)), key=lambda k: res[k])[::-1]
            pred_assers = list(user_emb_df.loc[l_idx,:].user_id.values)
            pred_assers = l_idx
  
            cur_true = [0 for i in range(N)]
            cur_true[row['next_user'][0]] = 1
            cur_pred = [0 for i in range(N)]
            for item_idx in range(20):
                cur_pred[pred_assers[item_idx]] = 20-item_idx
            pred_assers_list.append(pred_assers)
            y_true.append(cur_true)
            y_pred.append(res)

        asser_emb_df['pred_assers'] = pred_assers_list
        asser_emb_df['y_true'] = y_true
        asser_emb_df['y_pred'] = y_pred
        
     

        total_cnt_user = 0

        recall_10_list_user = []
        recall_20_list_user = []
        recall_30_list_user = []
        recall_50_list_user = []
        recall_100_list_user = []
        recall_150_list_user = []
        for idx, row in asser_emb_df.iterrows():

            total_cnt_user += 1
            recall_10_list_user.append(len(set(row['pred_assers'][0:10]).intersection(set(row['next_user'])))/min(10, len(set(row['next_user']))))
            recall_20_list_user.append(len(set(row['pred_assers'][0:20]).intersection(set(row['next_user'])))/min(20, len(set(row['next_user']))))
            recall_30_list_user.append(len(set(row['pred_assers'][0:30]).intersection(set(row['next_user'])))/min(30, len(set(row['next_user']))))
            recall_50_list_user.append(len(set(row['pred_assers'][0:50]).intersection(set(row['next_user'])))/min(50, len(set(row['next_user']))))
            recall_100_list_user.append(len(set(row['pred_assers'][0:100]).intersection(set(row['next_user'])))/min(100, len(set(row['next_user']))))
            recall_150_list_user.append(len(set(row['pred_assers'][0:150]).intersection(set(row['next_user'])))/min(150, len(set(row['next_user']))))
 
        avg_recall_10_user = statistics.mean(recall_10_list_user)
        avg_recall_20_user = statistics.mean(recall_20_list_user)
        avg_recall_30_user = statistics.mean(recall_30_list_user)
        avg_recall_50_user = statistics.mean(recall_50_list_user)
        avg_recall_100_user = statistics.mean(recall_100_list_user)
        avg_recall_150_user = statistics.mean(recall_150_list_user)
        y_true_array = np.asarray(list(asser_emb_df.y_true.values))
        y_pred_array = np.asarray(list(asser_emb_df.y_pred.values))
        ndcg_10 = ndcg_score(y_true_array, y_pred_array, k=10)
        ndcg_20 = ndcg_score(y_true_array, y_pred_array, k=20)
        ndcg_30 = ndcg_score(y_true_array, y_pred_array, k=30)
        ndcg_50 = ndcg_score(y_true_array, y_pred_array, k=50)
        ndcg_100 = ndcg_score(y_true_array, y_pred_array, k=100)
        ndcg_150 = ndcg_score(y_true_array, y_pred_array, k=150)
        ndcg = ndcg_score(y_true_array, y_pred_array)
        print('recall',avg_recall_10_user,avg_recall_20_user,avg_recall_30_user,avg_recall_50_user,avg_recall_100_user,avg_recall_150_user)
        print('ndcg',ndcg_10,ndcg_20,ndcg_30,ndcg_50,ndcg_100,ndcg_150,ndcg)
    

    print("after propogation")
    seq_future_pred(embedding,seq_embedding,meta,user_map,assertion_map,test)



main()