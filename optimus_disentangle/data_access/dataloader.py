"""
Data loader for STS tasks.
"""
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
import numpy as np


# read input dataset.
class SetSizeLineByLineTextDataset(Dataset):
    """
        Same as `LineByLineTextDataset` by Huggingface but modified to used fixed length sequences & to cache the result.
    """
    def __init__(
            self, tokenizer: PreTrainedTokenizer, file_path: str, set_seq_size, overwrite_cache=False, local_rank=-1, inject=None, disentangle=None, task="recon", srl_vocab=None
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
            # max_len = 0
            # num_50, num_40, num_30, num_20 = 0, 0, 0, 0

            if task == 'recon':
                print("reconstruction task")
                seq_texts = self._get_text_sequences_rec(file_path)

                # random.shuffle(seq_texts)
                self.examples = []

                # change here !!!!!!!!!!
                for text in seq_texts:
                    # remove blank space.
                    t0 = text[0].strip().split(" ")
                    label = text[1].strip().split(" ")
                    # t1 = [i for i in text[1].split(" ") if i not in (" ", ',')]
                    # t2 = [i for i in text[2].split(" ") if i not in (" ", ',')]

                    text_0 = ' '.join(t0) # </s> is the special token in T5.
                    text_1 = '</s> ' + ' '.join(t0)
                    text_2 = ' '.join(t0) + ' </s>'
                    text_0, text_1, text_2 = text_0.strip(), text_1.strip(), text_2.strip()

                    cur_len = len(tokenizer.encode(text_0))

                    if cur_len > set_seq_size:
                        continue

                    # T5 will split some word into multi-pieces, we should expand srl in the same way.
                    expand_label, dimention_inf, dimention_dict, srl_index = [], [], disentangle, []
                    if disentangle != None and srl_vocab != None:
                        tmp = [len(tokenizer.encode(i)) for i in text_0.split(' ')]
                        assert len(label) == len(tmp)
                        for i, t in enumerate(tmp):
                            # expand_label.extend([srl_vocab[label[i]]]*t)
                            expand_label.extend([srl_vocab[label[i]] if label[i] in dimention_dict else srl_vocab['FIX']]*t)
                            # replace each srl by value range.
                            srl_index.extend(label[i]*t if label[i] in dimention_dict else [dimention_dict["residual"]]*t)
                            dimention_inf.extend([dimention_dict[label[i]]]*t if label[i] in dimention_dict else [dimention_dict["residual"]]*t)

                        # specify PAD token
                        len_dim = len(dimention_inf)
                        remain_len = set_seq_size - len_dim
                        # tot_len = dimention_dict["residual"][1]
                        if remain_len > 0:
                            # dimention_inf.extend([(tot_len-1, tot_len)]*remain_len)
                            dimention_inf.extend([dimention_dict['residual']]*remain_len)
                            expand_label.extend([srl_vocab['PAD']]*remain_len)


                    token_text_0 = tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_1 = tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_2 = tokenizer.batch_encode_plus([text_2], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

                    self.examples.append({
                        'input_ids': token_text_0['input_ids'][0],
                        'label_ids': token_text_1['input_ids'][0],
                        'label1_ids': token_text_2['input_ids'][0],
                        'srl': torch.tensor(dimention_inf), # used in VQVAE
                        'srl_ids': torch.tensor(expand_label) # used in input embedding.
                    })

                start = time.time()
                # print("Max length is ", max_len)
                # print("num 50: {}, 40: {}, 30: {}, 20: {}".format(num_50, num_40, num_30, num_20))
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            elif task == 'inference_com':
                print("inference task with combined premises.")
                seq_texts = self._get_text_sequences(file_path)

                # random.shuffle(seq_texts)
                self.examples = []
                trigger = True
                max_len = 0

                # change here !!!!!!!!!!
                for text in seq_texts:
                    # remove blank space.
                    t0 = [i for i in text[0].split(" ") if i not in (" ", ',')]
                    t1 = [i for i in text[1].split(" ") if i not in (" ", ',')]
                    t2 = [i for i in text[2].split(" ") if i not in (" ", ',')]

                    if len(text) > 3:
                        if trigger:
                            print('using inference_type')
                            trigger = False

                        if inject == 'encoder_prefix':
                            # inference type as encoder input.
                            type = text[3]
                            text_0 = 'inference type is '+type+' </s> '+' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>' # </s> is the special token in T5.
                            text_1 = '</s> ' + ' '.join(t2)
                            text_2 = ' '.join(t2) + ' </s>'

                        elif inject == 'decoder_prefix':
                            # inference type as decoder prefix.
                            type = text[3]
                            text_0 = ' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>' # </s> is the special token in T5.
                            text_1 = '</s> ' + 'the inference type is ' + type + ' ' + ' '.join(t2)
                            text_2 = 'the inference type is '+type + ' ' + ' '.join(t2) + ' </s>'

                        elif inject == 'decoder_end':
                            # inference type as decoder end.
                            type = text[3]
                            text_0 = ' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>' # </s> is the special token in T5.
                            text_1 = '</s> ' + ' '.join(t2) + ' . ' + 'the inference type is ' + type
                            text_2 = ' '.join(t2) + ' . ' + 'the inference type is ' + type + ' </s>'

                        else:
                            exit("Error: wrong inject name.")

                    else:
                        if trigger:
                            print('do not using inference type')
                            trigger = False

                        text_0 = ' '.join(t0) + ' </s> ' + ' '.join(t1) + ' </s>' # </s> is the special token in T5.
                        text_1 = '</s> ' + ' '.join(t2)
                        text_2 = ' '.join(t2) + ' </s>'

                        cur_len = len(tokenizer.encode(text_0))
                        max_len = cur_len if cur_len > max_len else max_len

                    token_text_0 = tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_1 = tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_2 = tokenizer.batch_encode_plus([text_2], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

                    self.examples.append({'input_ids': token_text_0['input_ids'][0], 'label_ids': token_text_1['input_ids'][0], 'label1_ids': token_text_2['input_ids'][0]})

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            elif task == 'inference_sep':
                print("inference task with separated premises.")
                seq_texts = self._get_text_sequences(file_path)

                # random.shuffle(seq_texts)
                self.examples = []
                trigger = True
                max_len = 0
                max_len1 = 0

                # change here !!!!!!!!!!
                for text in seq_texts:
                    # remove blank space.
                    t0 = [i for i in text[0].split(" ") if i not in (" ", ',')]
                    t1 = [i for i in text[1].split(" ") if i not in (" ", ',')]
                    t2 = [i for i in text[2].split(" ") if i not in (" ", ',')]

                    if trigger:
                        print('do not using inference type')
                        trigger = False

                    text_0 = ' '.join(t0) # </s> is the special token in T5.
                    text_1 = '</s> ' + ' '.join(t2)
                    text_2 = ' '.join(t2) + ' </s>'
                    text_3 = ' '.join(t1)

                    token_text_0 = tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_1 = tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_2 = tokenizer.batch_encode_plus([text_2], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                    token_text_3 = tokenizer.batch_encode_plus([text_3], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

                    self.examples.append({'input_ids': token_text_0['input_ids'][0], 'label_ids': token_text_1['input_ids'][0],
                                          'label1_ids': token_text_2['input_ids'][0], 'input1_ids': token_text_3['input_ids'][0]})

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            else:
                exit("Error: wrong task name only support: 1. recon, 2. inference_com, 3. inference_sep")


    @staticmethod
    def _get_text_sequences(file_path):
        # read our dataset.
        dat = pd.read_csv(file_path, index_col=[0])
        seq_texts = dat.values.tolist()

        return seq_texts

    @staticmethod
    def _get_text_sequences_rec(file_path):
        dat = pd.read_csv(file_path, sep="&", header=None,
                          names=["text", "role"])
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


def get_dataset(args, tokenizer, set_seq_size, local_rank=-1, inject=None, disentangle=None, task="recon", srl_vocab=None):
    tr_file_path = args.train_data_file
    te_file_path = args.test_data_file
    return (SetSizeLineByLineTextDataset(tokenizer=tokenizer, file_path=tr_file_path, set_seq_size=set_seq_size, overwrite_cache=args.overwrite_cache, local_rank=local_rank, inject=inject, disentangle=disentangle, task=task, srl_vocab=srl_vocab),
            SetSizeLineByLineTextDataset(tokenizer=tokenizer, file_path=te_file_path, set_seq_size=set_seq_size, overwrite_cache=args.overwrite_cache, local_rank=local_rank, inject=inject, disentangle=disentangle, task=task, srl_vocab=srl_vocab))


def get_dataloader(args, train_dataset):
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size * 1, # self.args.n_gpu,
        sampler=RandomSampler(train_dataset),
        # collate_fn=self.data_collator,
    )

    return data_loader


def _get_dataset(args, tokenizer, set_seq_size):
    tr_file_path = args.train_data_file
    te_file_path = args.test_data_file

    dat = pd.read_csv(tr_file_path, sep="&", header=None, names=["text", "role"])
    seq_texts = dat.values.tolist()

    tr_x_examples = []
    tr_y_examples = []
    for text in seq_texts:
        # remove blank space.
        t0 = text[0].strip().split(" ")
        label = text[1].strip().split(" ")

        text_0 = '</s> ' + ' '.join(t0) # </s> is the special token in T5.
        text_0 = text_0.strip()
        text_1 = ' '.join(t0) + ' </s>' # </s> is the special token in T5.
        text_1 = text_1.strip()

        cur_len = len(tokenizer.encode(text_0))

        if cur_len > set_seq_size:
            continue

        token_text_0 = tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        token_text_1 = tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        tr_x_examples.append(token_text_0['input_ids'][0].tolist())
        tr_y_examples.append(token_text_1['input_ids'][0].tolist())

    dat = pd.read_csv(te_file_path, sep="&", header=None, names=["text", "role"])
    seq_texts = dat.values.tolist()

    te_x_examples = []
    te_y_examples = []
    for text in seq_texts:
        # remove blank space.
        t0 = text[0].strip().split(" ")
        label = text[1].strip().split(" ")

        text_0 = '</s> ' + ' '.join(t0) # </s> is the special token in T5.
        text_0 = text_0.strip()
        text_1 = ' '.join(t0) + ' </s>' # </s> is the special token in T5.
        text_1 = text_1.strip()

        cur_len = len(tokenizer.encode(text_0))

        if cur_len > set_seq_size:
            continue

        token_text_0 = tokenizer.batch_encode_plus([text_0], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        te_x_examples.append(token_text_0['input_ids'][0].tolist())
        token_text_1 = tokenizer.batch_encode_plus([text_1], max_length=set_seq_size, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        te_y_examples.append(token_text_1['input_ids'][0].tolist())

    return np.array(tr_x_examples), np.array(te_x_examples), np.array(tr_y_examples), np.array(te_y_examples)


if __name__ == '__main__':
    pass