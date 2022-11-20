import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def getIMDB(dev_size=0.1):
    ds = load_dataset('./dataset/imdb.py', split='train')
    test_data = load_dataset('./dataset/imdb.py', split='test')
    train_data, dev_data = ds.train_test_split(test_size=dev_size, seed=42).values()

    for data in train_data:
        data['aug_text'] = " [SEP] ".join(data['text'].split(" "))

    return train_data, dev_data, test_data

def collate_fn(batch,tokenizer,train=False):
    processed_batch = {}
    batch_size = len(batch)
    if train:
        texts,augs = [],[]
        for sample in batch:
            texts.append(sample['ori_input'])
            augs.append(sample['aug_input'])
            # batch_token = tokenizer(sum([sents, result1, result2], []), padding='longest', return_tensors='pt', return_token_type_ids=True)

        max_length = tokenizer(texts, padding='longest', return_tensors='pt')['input_ids'].size(1)*2
        batch_token = tokenizer(sum([texts, augs], []),
                                max_length=max_length, truncation=True, padding='max_length', return_tensors='pt',
                                return_token_type_ids=True)
        # retokenize
        processed_batch['ori_inputs'] = {'input_ids': batch_token['input_ids'][:batch_size],
                                          'attention_mask': batch_token['attention_mask'][:batch_size],
                                          'token_type_ids': batch_token['token_type_ids'][:batch_size]}
        processed_batch['aug_inputs'] = {'input_ids': batch_token['input_ids'][batch_size:],
                                          'attention_mask': batch_token['attention_mask'][batch_size:],
                                          'token_type_ids': batch_token['token_type_ids'][batch_size:]}
        processed_batch['labels'] = torch.tensor([sample['label'] for sample in batch])
    else:
        sents = [sample['ori_input'] for sample in batch]
        processed_batch['inputs'] = tokenizer(sents, padding='longest', return_tensors='pt', return_token_type_ids=True)
        processed_batch['labels'] = torch.tensor([sample['label'] for sample in batch])
    return processed_batch



class IMDBDataset(Dataset):
    def __init__(self, dataset,train=False):
        super(IMDBDataset, self).__init__()

        self.train = train
        self.labels = [data['label'] for data in dataset]
        self.labels = torch.Tensor(self.labels)

        self.text = [data['text'] for data in dataset]

        if train:
            self.aug_text = [" [SEP] ".join(data['text'].split(" ")) for data in dataset]

    def __getitem__(self, index):
        if self.train:
            return {
                'label': self.labels[index],
                'ori_input': self.text[index],
                'aug_input': self.aug_text[index],
            }
        else:
            return {
                'label': self.labels[index],
                'ori_input': self.text[index],
            }

    def __len__(self):
        return len(self.labels)
