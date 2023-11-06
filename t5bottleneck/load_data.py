from python_transformers import PreTrainedTokenizer
from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import random
import pickle
import time
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler


# read input dataset.
class SetSizeLineByLineTextDataset(Dataset):
    """
        Same as `LineByLineTextDataset` by Huggingface but modified to used fixed length sequences & to cache the result.
    """
    def __init__(
            self, tokenizer: PreTrainedTokenizer, file_path: str, set_seq_size, overwrite_cache=False
    ):
        print(f"Loading text.")

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f"cached_set_size_line_by_line_{tokenizer.__class__.__name__}_set_seq_size_{set_seq_size}_{filename}")

        if os.path.exists(cached_features_file) and not overwrite_cache:
            start = time.time()
            print(f"Loading features from cached file {cached_features_file}...")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            print("[took %.3f s]", time.time() - start)

        else:
            if not os.path.isfile(file_path):
                raise Exception(f"Can't find true file:\n{file_path}\nAlso can't find cahced file:\n{cached_features_file}")
            # don't create cache when running in parallel

            print(f"Creating features from dataset file at {directory}")

            seq_texts = self._get_text_sequences(file_path)

            random.shuffle(seq_texts)
            self.examples = []
            max_len = 0

            # change here !!!!!!!!!!
            for text in seq_texts:
                p = text[0]
                c = text[1]
                text_0 = p # </s> is the special token in T5.
                text_1 = '</s> ' + c
                text_2 = c + ' </s>'
                if len(tokenizer.encode(p)) > max_len:
                    max_len = len(tokenizer.encode(p))

                if '⁇' in tokenizer.decode(tokenizer.encode(p)).split(' '):
                    print('#####')
                    print(p)
                    print(tokenizer.encode(p))
                    print(tokenizer.decode(tokenizer.encode(p)))


                token_text_0 = tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                token_text_1 = tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                token_text_2 = tokenizer.batch_encode_plus([text_2], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

                self.examples.append({'input_ids': token_text_0['input_ids'][0], 'label_ids': token_text_1['input_ids'][0], 'label1_ids': token_text_2['input_ids'][0]})

            start = time.time()
            print("max_len: ", max_len)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    @staticmethod
    def _get_text_sequences(file_path):
        dat = pd.read_csv(file_path, sep="&", header=None, names=["input", "output"])
        seq_texts = dat.values.tolist()
        return seq_texts

    @staticmethod
    def _pad_tokens(set_size, tokens_tensor, pad_token):
        padedd = torch.ones(set_size, dtype=torch.long) * pad_token
        padedd[:tokens_tensor.size(0)] = tokens_tensor
        return padedd

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def get_dataset(args, tokenizer, set_seq_size):
    tr_file_path = args.train_data_file
    te_file_path = args.test_data_file
    return (SetSizeLineByLineTextDataset(tokenizer=tokenizer, file_path=tr_file_path, set_seq_size=set_seq_size, overwrite_cache=args.overwrite_cache),
            SetSizeLineByLineTextDataset(tokenizer=tokenizer, file_path=te_file_path, set_seq_size=set_seq_size, overwrite_cache=args.overwrite_cache))


def get_dataloader(args, train_dataset):
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size * 1,
        sampler=RandomSampler(train_dataset)
    )

    return data_loader


def _get_text_sequences(file_path):
    dat = pd.read_csv(file_path, sep="&", header=None, names=["input", "output"])
    seq_texts = dat.values.tolist()
    return seq_texts

if __name__ == '__main__':
    pass