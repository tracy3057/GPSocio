from torch.utils.data import Dataset
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
import itertools

class GPSocioTrainDataset(Dataset):
    def __init__(self, user2train, collator: FinetuneDataCollatorWithPadding):

        '''
        user2train: dict of sequence data, user--> item sequence
        '''
        
        self.user2train = user2train
        self.collator = collator
        self.users = sorted(user2train.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self.user2train[user]
        
        return seq

    def collate_fn(self, data):

        # return self.collator([{'items': line} for line in data])
        return self.collator([{'items': list(itertools.chain.from_iterable(line))} if isinstance(line[0], list) else {'items': line} for line in data[1]])



class GPSocioEvalDataset(Dataset):
    def __init__(self, user2train, user2val, user2test, mode, collator: EvalDataCollatorWithPadding):
        self.user2train = user2train
        self.user2val = user2val
        self.user2test = user2test
        self.collator = collator

        if mode == "val":
            self.users = list(self.user2val.keys())
        else:
            self.users = list(self.user2test.keys())

        self.mode = mode

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.user2train[user] if self.mode == "val" else self.user2train[user] + self.user2val[user]
        label = self.user2val[user] if self.mode == "val" else self.user2test[user]
        
        return seq, label

    def collate_fn(self, data):

        # data_sample = []
        # for item in data:
        #     data_sample.append(item[1])

        # return self.collator([{'items': list(itertools.chain.from_iterable(line))} if isinstance(line[0], list) else {'items': line} for line in data_sample])
        return self.collator([{'items': line[0], 'label': line[1]} for line in data])
