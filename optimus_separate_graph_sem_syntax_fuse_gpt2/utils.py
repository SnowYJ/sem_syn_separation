import math
import random
import numpy as np
import torch
import contextlib
import joblib
from typing import Iterable
from saf import Sentence
from transformers import PreTrainedTokenizer


def set_seed(seed):  # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]


def load_sent(path, n=None):
    sents = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(';')
            d = {'term': parts[2], 'tokens': [], 'roles': []}
            for s in parts[4:]:
                sp = s.split('/')
                srl = sp[-1]
                phrase = ''.join(sp[:-1])
                phrase_tokens = phrase.split()
                d['tokens'] += phrase_tokens
                d['roles'] += [srl] * len(phrase_tokens)
            sents.append(d)
            if n:
                n -= 1
                if not n:
                    return sents
    return sents


def load_sample_def(corpus: Iterable[Sentence], split: tuple = (.7, .2), randomize: bool = True):
    sents = list(corpus)
    if (randomize):
        random.shuffle(sents)

    train_cut = int(split[0] * len(sents))
    valid_cut = int((split[0] + split[1]) * len(sents))
    train, valid = sents[0:train_cut], sents[train_cut: valid_cut]
    test = sents[valid_cut:] if (split[0] + split[1] < 1) else []

    return train, valid, test


def load_sample(corpus: Iterable[Sentence], split: tuple = (.7, .2), randomize: bool = True):
    """
    this is used in the work VQVAE. train = [:12000] valid = [12000:]
    """
    sents = list(corpus)
    train, valid = sents[0:12000], sents[12000:]
    return train, valid, valid


def load_sample_others(corpus: Iterable[Sentence], split: tuple = (.9, .1), randomize: bool = True):
    sents = list(corpus)
    if (randomize):
        random.shuffle(sents)

    train_cut = int(split[0] * len(sents))
    valid_cut = int((split[0] + split[1]) * len(sents))
    train, valid = sents[0:train_cut], sents[train_cut: valid_cut]

    return train, valid, valid


def load_sample_inf(corpus: Iterable[Sentence], split: tuple = (.7, .2), randomize: bool = True):
    sents = list(corpus)
    train, valid = sents[0:5000], sents[5000:]
    return train, valid, valid


def conv_sent_dict(sent: Sentence, emb_tokenizer: PreTrainedTokenizer = None, decode_tokenizer: PreTrainedTokenizer=None, model=None, args=None):
    sent_dict = None
    if args.exp in ['exp_infer', 'exp_cvae']:
        if args.inference_premises_com:
            p1 = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.tokenize(sent['premises'][0]))
            p2 = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.tokenize(sent['premises'][1]))
            c = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.tokenize(sent['conclusion']))
            p1_p2 = emb_tokenizer.add_special_tokens_sentences_pair(p1, p2)
            gpt2_bos = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.bos_token)
            gpt2_eos = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.eos_token)
            c = [gpt2_bos] + c + [gpt2_eos]

            bert_pad = emb_tokenizer.pad_token_id
            gpt_pad = decode_tokenizer.pad_token_id

            sent_dict = {'def_codes': p1_p2, 'conclusion': c, 'bert_pad': bert_pad, 'gpt_pad': gpt_pad}

        elif args.inference_premises_sep:
            p1 = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.tokenize(sent['premises'][0]))
            p2 = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.tokenize(sent['premises'][1]))
            c = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.tokenize(sent['conclusion']))
            gpt2_bos = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.bos_token)
            gpt2_eos = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.eos_token)
            bert_cls = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.cls_token)
            bert_sep = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.sep_token)
            gpt_pad = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.pad_token)
            bert_pad = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.pad_token)

            c = [gpt2_bos] + c + [gpt2_eos]
            p1 = [bert_cls] + p1 + [bert_sep]
            p2 = [bert_cls] + p2 + [bert_sep]

            sent_dict = {'def_codes': p1, 'premise2': p2, 'conclusion': c, 'gpt_pad': gpt_pad, 'bert_pad': bert_pad}
        else:
            exit('Error when running in conv_sent_dict func.')
    else:
        if emb_tokenizer:
            if model in ['optimus', 'conditional_optimus', 'DSVAE']:
                term_inputs = emb_tokenizer.encode(sent.annotations["definiendum"]) if "definiendum" in sent.annotations.keys() else None
                term = sent.annotations["definiendum"] if "definiendum" in sent.annotations.keys() else 'term'

                # use bert tokenizer and GPT tokenizer. It will add special token '#, G' automatically for a sentence.
                bert_list = emb_tokenizer.tokenize(' '.join([t.surface for t in sent.tokens]))
                gpt_list = decode_tokenizer.tokenize(' '.join([t.surface for t in sent.tokens]))

                def_inputs = torch.IntTensor(emb_tokenizer.convert_tokens_to_ids(bert_list))
                def_gpt_inputs = torch.IntTensor(decode_tokenizer.convert_tokens_to_ids(gpt_list))

                roles, roles1 = list(), list()
                bert_tmp_list = [emb_tokenizer.tokenize(t.surface) for t in sent.tokens]
                gpt_tmp_list = [decode_tokenizer.tokenize(t.surface) for t in sent.tokens]

                # after tokenize, some words might be splited into several parts.
                if 'DSR' in sent.tokens[0].annotations:
                    for (i, token) in enumerate(sent.tokens):
                        roles1.extend([token.annotations["DSR"]] * len(gpt_tmp_list[i]))
                        roles.extend([token.annotations['DSR']] * len(bert_tmp_list[i]))

                    assert len(roles) == len(def_inputs)
                    assert len(roles1) == len(def_gpt_inputs)
                else:
                    pass

                sent_dict = {"term": term, "tokens": [token.surface for token in sent.tokens], "roles": roles,
                             "roles1": roles1, "term_codes": term_inputs, "def_codes": def_inputs,
                             "def1_codes": def_gpt_inputs,
                             "bert_cls": emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.cls_token),
                             "bert_sep": emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.sep_token),
                             'gpt_bos': decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.bos_token),
                             'gpt_eos': decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.eos_token),
                             'gpt_pad': decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.pad_token),
                             'bert_pad': emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.pad_token)}

            elif model == 'TransformerCVAE':
                # input_list = emb_tokenizer.tokenize(' '.join([t.surface for t in sent.tokens]))
                input_list = emb_tokenizer.tokenize(' '.join(['start']+[t.surface for t in sent.tokens]))
                gpt_inputs = torch.IntTensor(emb_tokenizer.convert_tokens_to_ids(input_list))[1:]
                roles = list()
                # gpt_tmp_list = [emb_tokenizer.tokenize(t.surface) for t in sent.tokens]

                # # using gpt2 tokenizer directly.
                # roles_input_list = emb_tokenizer.tokenize(' '.join([t.annotations['DSR'] for t in sent.tokens]))
                # role_inputs = torch.IntTensor(emb_tokenizer.convert_tokens_to_ids(roles_input_list))
                for (i, token) in enumerate(sent.tokens):
                    # roles.extend([token.annotations['DSR']] * len(gpt_tmp_list[i]))
                    roles.extend([token.annotations['DSR']] * 1)

                sent_dict = {"term": sent.annotations["definiendum"],
                             "tokens": [token.surface for token in sent.tokens],
                             "roles": roles,
                             "term_codes": input_list,
                             "def_codes": gpt_inputs,
                             "gpt_bos": emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.bos_token),
                             "gpt_eos": emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.eos_token),
                             "gpt_pad": emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.pad_token),
                             "gpt_sep": emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.sep_token)
                             }
            else:
                # rnn
                term_inputs = emb_tokenizer(sent.annotations["definiendum"], return_tensors="pt").input_ids.int()
                def_tk_list = [emb_tokenizer(token.surface, return_tensors="pt").input_ids[0][:-1].int() for token in sent.tokens]
                def_inputs = torch.IntTensor(torch.cat(def_tk_list))
                roles = list()
                """
                T5 tokenizer could chop the words. so, the size of each word might be bigger than 1.
                """
                for (i, token) in enumerate(sent.tokens):
                    roles.extend([token.annotations["DSR"]] * len(def_tk_list[i]))

                sent_dict = {"term": sent.annotations["definiendum"],
                             "tokens": [token.surface for token in sent.tokens],
                             "roles": roles,
                             "term_codes": term_inputs[0, :-1],
                             "def_codes": def_inputs,
                             "eos": emb_tokenizer("a", return_tensors="pt").input_ids[0][-1].item()}
        else:
            sent_dict = {"term": sent.annotations["definiendum"],
                         "tokens": [token.surface for token in sent.tokens],
                         "roles": [token.annotations["DSR"] for token in sent.tokens]}

    return sent_dict


def write_sent(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')


def write_doc(docs, path):
    with open(path, 'w') as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')


def write_z(z, path):
    with open(path, 'w') as f:
        for zi in z:
            for zij in zi:
                f.write('%f ' % zij)
            f.write('\n')


def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')


def lerp(t, p, q):
    return (1 - t) * p + t * q


# spherical interpolation https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
def slerp(t, p, q):
    o = np.arccos(np.dot(p / np.linalg.norm(p), q / np.linalg.norm(q)))
    so = np.sin(o)
    return np.sin((1 - t) * o) / so * p + np.sin(t * o) / so * q


def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0 * i / (n - 1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)


def convert(seconds):
    min_, sec = divmod(seconds, 60)
    hour, min_ = divmod(min_, 60)
    return "%dh:%02dm:%02ds" % (hour, min_, sec)


millnames = ['', ' Thousands', ' Millions', ' Billions', ' Trillions']


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()
