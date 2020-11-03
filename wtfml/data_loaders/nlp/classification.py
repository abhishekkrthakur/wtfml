import torch

class classificationDataset:
    def __init__(self,text,target,tokenizer,max_len:int,special_token:bool):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_token = special_token

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self,item):
        text = str(self.text[item])
        target = str(self.target[item])

        inputs = self.tokenizer.encode_plus(
            text,
            target,
            max_len = self.max_len,
            add_special_token =self.special_token 
        )

        ids = inputs["ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        padding_length = self.max_len-len(ids)

        ids = ids + ([0] * padding_length)
        token_type_ids = token_type_ids +( [0] * padding_length)
        mask = mask + ([0] * padding_length)

        return {
            "ids":torch.tensor(ids,dtype=torch.long),
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),
            "mask":torch.tensor(mask,dtype=torch.long),
            "targets":torch.tensor(self.target,dtype=torch.float),
        }


class classificationDataLoader:
    def __init__(self,text,target,tokenizer,max_len:int,special_token:bool):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_token = special_token
        self.dataset = classificationDataset(
            text = self.text,
            target = self.target,
            tokenizer=self.tokenizer,
            max_len = self.max_len,
            special_token = self.special_token
        )

    def fetch(self,batch_size,num_workers,shuffle=True,sampler=None,tpu=False):
        if tpu:
            sampler = torch.utils.data.DistributedSampler(
                self.dataset,num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle)
        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
        return data_loader