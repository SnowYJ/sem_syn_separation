import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def get_first_ix(terms, vocab):
    for w in terms:  # check for single words
        if w in vocab.word2idx:
            return vocab.word2idx[w]
    # for w in term_:  # check for tokens of composed terms e.g. active_citizen
    #     if len(w.split('_')) > 1:
    #         for t in w.split('_'):
    #             if t in vocab.word2idx:
    #                 return vocab.word2idx[t]
    return vocab.unk


# def get_rec_batch(x, vocab, device):
#     x_batch, y_batch = [], []
#     max_len = max([len(s["def_codes"]) for s in x])
#
#     for i, s in enumerate(x):
#         x_ix = s["def_codes"]
#         pad = torch.IntTensor([vocab.pad] * (max_len - len(x_ix) + 1))
#         x_batch.append(torch.cat([x_ix, pad])[None, :])
#         batch = torch.IntTensor(torch.cat(x_batch, dim=0)).t().contiguous()
#
#     return batch.to(device), batch.detach().clone().to(device)


def get_rec_batch(x, vocab, device, model, t0=None, t1=None, infer_type=None):
    go_x, x_eos = [], []
    max_len = max([len(s["def_codes"]) for s in x])

    if model == 'optimus':
        # to avoid gpt2 ignore the [eos] token, so length plus 1
        max_len2 = max([len(s["def1_codes"]) for s in x])+1
        for s in x:
            s_bert_idx, s_gpt_idx = s["def_codes"].tolist(), s["def1_codes"].tolist()
            bert_padding, gpt_padding = [s['bert_pad']] * (max_len - len(s["def_codes"])), \
                                        [s['gpt_pad']] * (max_len2 - len(s["def1_codes"]))

            bert_cls, bert_sep = s['bert_cls'], s['bert_sep']
            gpt_bos, gpt_eos = s['gpt_bos'], s['gpt_eos']

            go_x.append([bert_cls] + s_bert_idx + [bert_sep] + bert_padding)
            x_eos.append([gpt_bos] + s_gpt_idx + [gpt_eos] + gpt_padding)
    else:
        exit()

    return torch.IntTensor(go_x).t().contiguous().to(device), \
           torch.IntTensor(x_eos).t().contiguous().to(device)


def get_batch_dm(x, vocab, device, model, t0=None, t1=None, infer_type=None):
    """
    Batch for training Definition Model.
    For example:
    - burnish -> enhance or improve.
    - x: burnish enhance or improve
    - y: enhance or improve </s>
    """
    x_batch, y_batch = [], []
    if model == 'optimus':
        max_len_def_x = max([len(s["term_codes"]) for s in x])
        max_len_def_bert = max([len(s["def_codes"]) for s in x])
        max_len_def_gpt2 = max([len(s["def1_codes"]) for s in x])

        for i, s in enumerate(x):
            gpt_bos, gpt_eos = [s['gpt_bos']], [s['gpt_eos']]
            x_ix = s["term_codes"] + s["def_codes"].tolist()
            y_ix = s["def1_codes"].tolist()

            pad_x = [s['bert_pad']] * (max_len_def_x + max_len_def_bert - len(x_ix) + 1)
            pad_y = [s['gpt_pad']] * (max_len_def_gpt2 - len(y_ix) + 1)
            x_batch.append(x_ix + pad_x)
            y_batch.append(gpt_bos + y_ix + gpt_eos + pad_y)
    else:
        exit()

    return torch.IntTensor(x_batch).t().contiguous().to(device), \
           torch.IntTensor(y_batch).t().contiguous().to(device)


def get_batch_w(x, vocab, device, model, t0=None, t1=None, infer_type=None):
    """
    Batch for generating word definition: single words + embedding
    """
    x_batch, y_batch = [], []
    if model == 'optimus':
        max_len_def_x = max([len(s["term_codes"]) for s in x])
        max_len_def_bert = max([len(s["def_codes"]) for s in x])
        max_len_def_gpt2 = max([len(s["def1_codes"]) for s in x])

        for i, s in enumerate(x):
            gpt_bos, gpt_eos = [s['gpt_bos']], [s['gpt_eos']]
            x_ix = s["term_codes"] # + s["def_codes"].tolist()
            y_ix = s["def1_codes"].tolist()

            pad_x = [s['bert_pad']] * (max_len_def_x + max_len_def_bert - len(x_ix) + 1)
            pad_y = [s['gpt_pad']] * (max_len_def_gpt2 - len(y_ix) + 1)
            x_batch.append(x_ix + pad_x)
            y_batch.append(gpt_bos + y_ix + gpt_eos + pad_y)
    else:
        exit()

    return torch.IntTensor(x_batch).t().contiguous().to(device), torch.IntTensor(y_batch).t().contiguous().to(device)


def get_batch_w_sc(x, vocab, device):
    """
    Batch of single words + vector (for the task of generating their definitions)
    """
    x_batch = []
    max_len_def_x = max([len(s["term_codes"]) for s in x])
    max_len_def_y = max([len(s["def_codes"]) for s in x])

    for i, s in enumerate(x):
        x_ix = torch.cat([s["term_codes"], s["def_codes"]])
        pad_x = torch.IntTensor([vocab.pad] * (max_len_def_x + max_len_def_y - len(x_ix) + 1))
        x_batch.append(torch.cat([x_ix, pad_x])[None, :])

    return torch.IntTensor(torch.cat(x_batch, dim=0)).t().contiguous().to(device)


def get_batch_roles(x, vocab, device, model, tokenizer_encoder, tokenizer_decoder, infer_type=None):
    """
        Exp2:  batch for training reconstruction with roles.
        input1: go + x, target: x + eos
        input2: go + role, target: role + eos
    """
    go_x, x_eos, go_x1, x1_eos = [], [], [], []
    max_len = max([len(s["def_codes"]) for s in x])
    if model in ['optimus']:
        max_len2 = max([len(s["def1_codes"]) for s in x])+1
        role_pad, role_bos, role_eos, role_unknown = vocab.role_size, vocab.role_size+1, vocab.role_size+2, vocab.role_size+3
        for s in x:
            s_bert_idx, s_gpt_idx = s["def_codes"].tolist(), s["def1_codes"].tolist()
            bert_padding, gpt_padding = [s['bert_pad']] * (max_len - len(s["def_codes"])), [s['gpt_pad']] * (max_len2 - len(s["def1_codes"]))
            bert_sr_idx = [vocab.role2idx[w] if w in vocab.role2idx else role_unknown for w in s['roles']]
            gpt_sr_idx = [vocab.role2idx[w] if w in vocab.role2idx else role_unknown for w in s['roles1']]

            role_bert_padding, role_gpt_padding = [role_pad] * (max_len - len(s['roles'])), [role_pad] * (max_len2 - len(s['roles1']))

            bert_cls, bert_sep, gpt_bos, gpt_eos = s['bert_cls'], s['bert_sep'], s['gpt_bos'], s['gpt_eos']

            go_x.append([bert_cls] + s_bert_idx + [bert_sep] + bert_padding)
            x_eos.append([gpt_bos] + s_gpt_idx + [gpt_eos] + gpt_padding)
            """
            role label is feed into a new embedding layer. So, do not use bert or gpt special tokens. instead, 
            """
            go_x1.append([role_bos] + bert_sr_idx + [role_eos] + role_bert_padding)
            x1_eos.append([role_bos] + gpt_sr_idx + [role_eos] + role_gpt_padding)

    elif model == 'conditional_optimus':
        max_len2 = max([len(s["def1_codes"]) for s in x])+1 # gpt 2
        max_role_len, max_role_len2 = max([len(s["roles"]) for s in x]), max([len(s["roles1"]) for s in x])

        for s in x:
            s_bert_idx, s_gpt_idx = s["def_codes"].tolist(), s["def1_codes"].tolist()
            ######################## encoder SRL input ########################
            temp_list = []
            for w in s['roles']:
                if w not in vocab.role2idx: w = 'ROLE-UNK'
                if w == 'O': w = 'O-ROLE'
                temp_list.append(w)
            bert_sr_idx = tokenizer_encoder.convert_tokens_to_ids(temp_list)
            ######################## decoder SRL input ########################
            temp_list1 = []
            for w in s['roles1']:
                if w not in vocab.role2idx: w = 'ROLE-UNK'
                if w == 'O': w = 'O-ROLE'
                temp_list1.append(w)
            gpt_sr_idx = tokenizer_decoder.convert_tokens_to_ids(temp_list1)
            #####################################################################
            bert_cls, bert_sep, gpt_bos, gpt_eos = s['bert_cls'], s['bert_sep'], s['gpt_bos'], s['gpt_eos']

            bert_padding = [s['bert_pad']] * (max_len + max_role_len - len(s["def_codes"]) - len(bert_sr_idx))
            gpt_padding = [s['gpt_pad']] * (max_len2 + max_role_len2 - len(s["def1_codes"]) - len(gpt_sr_idx))

            go_x.append([bert_cls] + s_bert_idx + [bert_sep] + bert_sr_idx + bert_padding)
            x_eos.append([gpt_bos] + s_gpt_idx + [gpt_eos] + gpt_sr_idx + [gpt_eos] + gpt_padding)

        return torch.IntTensor(go_x).t().contiguous().to(device), torch.IntTensor(x_eos).t().contiguous().to(device), 0, 0
    else:
        print('loading batch failure')
        exit()

    return torch.IntTensor(go_x).t().contiguous().to(device), \
           torch.IntTensor(x_eos).t().contiguous().to(device), \
           torch.IntTensor(go_x1).t().contiguous().to(device), \
           torch.IntTensor(x1_eos).t().contiguous().to(device)


def get_batch_infer(x, vocab, device, model, t0=None, t1=None, infer_type=None):
    """
    batch for entailmentbank inference.
    """
    if infer_type:
        # combination
        bert_pad = x[0]['bert_pad']
        gpt_pad  = x[0]['gpt_pad']
        input_ids_bert = pad_sequence([torch.tensor(f['def_codes'], dtype=torch.long) for f in x], batch_first=True, padding_value=bert_pad)
        input_ids_gpt = pad_sequence([torch.tensor(f['conclusion'], dtype=torch.long) for f in x], batch_first=True, padding_value=gpt_pad)

        return input_ids_bert.t().to(device), input_ids_gpt.t().to(device)
    else:
        # separation
        bert_pad = x[0]['bert_pad']
        gpt_pad  = x[0]['gpt_pad']

        input_ids_bert_0 = pad_sequence([torch.tensor(f['def_codes'], dtype=torch.long) for f in x], batch_first=True, padding_value=bert_pad)
        input_ids_bert_1 = pad_sequence([torch.tensor(f['premise2'], dtype=torch.long) for f in x], batch_first=True, padding_value=bert_pad)
        input_ids_gpt = pad_sequence([torch.tensor(f['conclusion'], dtype=torch.long) for f in x], batch_first=True, padding_value=gpt_pad)

        return input_ids_bert_0.t().to(device), input_ids_gpt.t().to(device), input_ids_bert_1.t().to(device)


def get_batches(data, vocab, batch_size, device, type_='exp1', model='rnn', tokenizer_encoder=None, tokenizer_decoder=None, infer_type=None):
    d = {'exp1': get_rec_batch,
         'exp2': get_batch_roles,
         'exp3': get_batch_roles,
         'exp3_2': get_batch_roles,
         'exp4_train': get_batch_dm,
         'exp4_gen': get_batch_w,
         'exp_infer': get_batch_infer}
    get_batch_ = d[type_]

    order = range(len(data))
    z_par = (order, data)
    z = sorted(zip(*z_par), key=lambda i: len(i[1]['def_codes'])) # zip(*z_par)
    order, data, = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_(data[i: j], vocab, device, model, tokenizer_encoder, tokenizer_decoder, infer_type))
        i = j

    return batches, order


def get_lm_batch(data, i, sent_size, batch_size, vocab, device):
    x, x1 = [], []

    for j in range(batch_size):
        ix = i + j + sent_size * j
        t = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in data[ix: ix + sent_size]]
        t1 = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in data[ix + 1: ix + sent_size + 1]]
        x.append(t)
        x1.append(t1)

    return torch.LongTensor(x).t().contiguous().to(device), \
           torch.LongTensor(x1).t().contiguous().to(device)


def get_batches_lm(file, vocab, batch_size, device, sent_size=30):
    with open(file, "r") as infile:
        data = infile.read().lower().split()
    batches = []
    i = 0
    while i < len(data) - ((sent_size + 1) * batch_size):
        batches.append(get_lm_batch(data, i, sent_size, batch_size, vocab, device))
        i += (sent_size + 1) * batch_size
    return batches
