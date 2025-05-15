import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def main(ratio):
    assertion_map = json.load(open("target_data/umap.json"))
    user_map = json.load(open("target_data/smap.json"))
    user_embedding = torch.load("target_data/preprocess/item_embeddings_target_data")
    asser_embedding_dict = json.load(open("target_data/preprocess/asser_embeddings_target_data.json"))
    train_org = json.load(open("target_data/train.json"))
    train = {}
    data = pd.read_pickle("target_data/selected_df_sequence_test.pkl")
    assertion_author_df = data[['assertion_id','author_id']].explode('author_id').reset_index(drop=True).groupby(['author_id'])['assertion_id'].apply(list).reset_index()
    assertion_author_df['test_asser'] = assertion_author_df['assertion_id'].apply(lambda x: x[-1]).astype(str)
    test_asser_mapped = []
    train_asser_mapped = []
    for idx, row in assertion_author_df.iterrows():
        test_asser_mapped.append(assertion_map[row['test_asser']])
        for item in row['assertion_id']:
            train_asser_mapped.append(assertion_map[str(item)])

    assertion_author_df['test_asser'] = test_asser_mapped
    assertion_author_df['assertion_id'] = assertion_author_df['assertion_id'].apply(lambda x: x[:-1])
    assertion_author_df = assertion_author_df.loc[assertion_author_df['assertion_id'].map(len)>=1]

    test_list = []
    for idx, row in assertion_author_df.iterrows():
        test_list.append((user_map[row['author_id']], str(row['test_asser'])))
    for key, value in train_org.items():
        new_value = []
        for user in value:
            if (user, key) in test_list:
                continue
            else:
                new_value.append(user)
        train[key] = new_value
        
    user_asser_map = {}
    asser_user_map = train
    asser_new_id_map = {}

    asser_cur_idx = 0
    for key, values in train.items():
        asser_new_id_map[key] = asser_cur_idx
        asser_cur_idx += 1
        for value in values:
            if value not in list(user_asser_map.keys()):
                user_asser_map[value] = [asser_new_id_map[key]]
            else:
                user_asser_map[value].append(asser_new_id_map[key])
    updated_user_embedding = user_embedding.tolist().copy()
    updated_asser_embedding = list(asser_embedding_dict.values()).copy()
    cur_user_embedding = user_embedding.tolist().copy()
    asser_embedding = list(asser_embedding_dict.values())
    cur_asser_embedding = asser_embedding.copy()

    for epoch in range(1):
        for key, value in user_asser_map.items():
            try:
                update_df = pd.DataFrame(list([cur_asser_embedding[i] for i in value]))
                delta_asser_embedding = list(update_df.mean().values)
                delta_asser_embedding = [ratio*i for i in delta_asser_embedding]
                updated_user_embedding[key] = [sum(i) for i in zip(cur_user_embedding[key], delta_asser_embedding)]
            except:
                updated_user_embedding[key] = cur_user_embedding[key]
        for key, value in asser_user_map.items():
            try:
                update_df = pd.DataFrame(list([cur_user_embedding[i] for i in value]))
                delta_user_embedding = list(update_df.mean().values)
                delta_user_embedding = [0*i for i in delta_user_embedding]

                updated_asser_embedding[asser_new_id_map[key]] = [sum(i) for i in zip(cur_asser_embedding[asser_new_id_map[key]], delta_user_embedding)]
            except:
                updated_asser_embedding[asser_new_id_map[key]] = cur_asser_embedding[asser_new_id_map[key]]

        cur_user_embedding = updated_user_embedding
        cur_asser_embedding = updated_asser_embedding

    updated_user_embedding = torch.FloatTensor(updated_user_embedding).to('cuda:0')
    asser_embedding_new = {}
    asser_idx = list(asser_embedding_dict.keys())
    for idx in range(len(cur_asser_embedding)):
        asser_embedding_new[asser_idx[idx]] = updated_asser_embedding[idx]

    torch.save(updated_user_embedding, "target_data/preprocess/propagated_item_embedding_asser_pred")
    json.dump(asser_embedding_dict,open("target_data/preprocess/propagated_asser_embedding_asser_pred.json","w"))

ratio = 3
main(ratio)