# read math reasoning dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import re


def MathInferenceCorpus(path):
    with open(path, 'r') as file:
        # Read the entire text
        data = file.readlines()
    examples = []
    for line in data:
        p, p1, c, c1 = line[:-1].split('&')
        example = {
            'premises': p,
            'premises_s': p1,
            'conclusion': c,
            'conclusion_s': c1
        }

        examples.append(example)

    return examples


def MathReconstructCorpus(path):
    with open(path, 'r') as file:
        # Read the entire text
        data = file.readlines()
    examples = []
    for line in data:
        p, p1 = line[:-1].split('&')
        example = {
            'premises': p,
            'premises_s': p1,
            'conclusion': p,
            'conclusion_s': p1
        }

        examples.append(example)

    return examples


def conv_sent_dict(sent, emb_tokenizer, decode_tokenizer, token_level):
    if token_level == 'subword':
        p = emb_tokenizer.convert_tokens_to_ids(emb_tokenizer.tokenize(sent['premises']))
        c = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.tokenize(sent['conclusion']))
    elif token_level == 'char_add_latex_tokens_without_var':
        p = sent['premises'].split('[SEP]')[0].strip()
        c = sent['conclusion'].split('[SEP]')[0].strip()
        latex_token = ["\\frac", "\\sin", "\\cos", "\\log", "e^"]

        if any([s in p for s in latex_token]):
            pattern = '|'.join(re.escape(token) for token in latex_token)
            tokenized_parts = re.split(f'({pattern})', p)
            result = []
            for part in tokenized_parts:
                if part in latex_token:
                    result.append(part)
                else:
                    result.extend(list(part))
            p = [item.strip() for item in result if item.strip() != '']
        else:
            p = [i.strip() for i in list(p) if i.strip() != '']

        if any([s in c for s in latex_token]):
            pattern = '|'.join(re.escape(token) for token in latex_token)
            tokenized_parts = re.split(f'({pattern})', c)
            result = []
            for part in tokenized_parts:
                if part in latex_token:
                    result.append(part)
                else:
                    result.extend(list(part))
            c = [item.strip() for item in result if item.strip() != '']
        else:
            c = [i.strip() for i in list(c) if i.strip() != '']

        p = emb_tokenizer.convert_tokens_to_ids(p)
        c = decode_tokenizer.convert_tokens_to_ids(c)
    else:
        # char-level
        c_p, c_c = [], []
        for i, s in enumerate(sent['premises'].split('[SEP]')):

            s = s.strip()
            if s in ['where', 'the', 'variable', 'is'] and token_level == 'char_for_latex_only':
                c_p += [s]
                continue

            c_p += [c for c in list(s) if c.strip() != '']
            if i == len(sent['premises'].split('[SEP]')) - 1:
                pass
            else:
                c_p += ['[SEP]']

        for i, s in enumerate(sent['conclusion'].split('[SEP]')):
            s = s.strip()
            if s in ['where', 'the', 'variable', 'is'] and token_level == 'char_for_latex_only':
                c_p += [s]
                continue

            c_c += [c for c in list(s) if c.strip() != '']
            if i == len(sent['conclusion'].split('[SEP]')) - 1:
                pass
            else:
                c_c += ['[SEP]']

        if token_level == 'char_for_latex_only':
            c_p = emb_tokenizer.tokenize(' '.join(c_p))
            c_c = decode_tokenizer.tokenize(' '.join(c_c))

        p = emb_tokenizer.convert_tokens_to_ids(c_p)
        c = decode_tokenizer.convert_tokens_to_ids(c_c)

    gpt2_bos = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.bos_token)
    gpt2_eos = decode_tokenizer.convert_tokens_to_ids(decode_tokenizer.eos_token)
    c = [gpt2_bos] + c + [gpt2_eos]

    bert_pad = emb_tokenizer.pad_token_id
    gpt_pad = decode_tokenizer.pad_token_id

    sent_dict = {'def_codes': p, 'conclusion': c, 'bert_pad': bert_pad, 'gpt_pad': gpt_pad}

    return sent_dict


def get_batch_infer(x, x_bow, x_struct, vocab, vocab_s, device, model_type):
    bert_pad = x[0]['bert_pad']
    gpt_pad = x[0]['gpt_pad']
    input_ids_bert = pad_sequence([torch.tensor(f['def_codes'], dtype=torch.long) for f in x], batch_first=True, padding_value=bert_pad)
    input_ids_gpt = pad_sequence([torch.tensor(f['conclusion'], dtype=torch.long) for f in x], batch_first=True, padding_value=gpt_pad)

    if model_type in ['lstm', 'gae']:
        input_ids_bert_bow, input_ids_gpt_bow = get_batch_bow(x_bow, vocab, device)
    else:
        input_ids_bert_bow, input_ids_gpt_bow = None, None

    if model_type == 'lstm':
        # lstm need padding during training
        input_ids_bert_s, input_ids_gpt_s = get_batch_struct(x_struct, vocab_s, device)
    else:
        # gae don't need
        input_ids_bert_s, input_ids_gpt_s = x_struct, x_struct

    return input_ids_bert.t().to(device), input_ids_gpt.t().to(device), \
           input_ids_bert_bow, input_ids_gpt_bow, \
           input_ids_bert_s, input_ids_gpt_s


def get_batch_bow(x, vocab, device):
    go_x, x_eos = [], []
    max_len_p = max([len(s['premise']) for s in x])
    max_len_c = max([len(s['conclusion']) for s in x])
    size = vocab.size

    for s in x:
        p_idx = [0 for _ in range(size)]
        for w in s['premise']:
            index = vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
            p_idx[index] += 1
        # p_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s['premise']]
        # padding = [vocab.pad] * (max_len_p - len(s['premise']))
        go_x.append(p_idx)

        c_idx = [0 for _ in range(size)]
        for w in s['conclusion']:
            index = vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
            c_idx[index] += 1
        # c_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s['conclusion']]
        # padding = [vocab.pad] * (max_len_c - len(s['conclusion']))
        x_eos.append(c_idx)

    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)


def get_batch_struct(x, vocab, device):
    go_x, x_eos = [], []
    max_len_p = max([len(s['premise']) for s in x])
    max_len_c = max([len(s['conclusion']) for s in x])

    for s in x:
        p_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s['premise']]
        padding = [vocab.pad] * (max_len_p - len(s['premise']))
        go_x.append(p_idx + padding)

        c_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s['conclusion']]
        padding = [vocab.pad] * (max_len_c - len(s['conclusion']))
        x_eos.append(c_idx + padding)

    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)


def get_batches(data, data_bow, data_struct, batch_size, vocab, vocab_s, device, model_type):
    order = range(len(data))
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_infer(data[i: j], data_bow[i:j], data_struct[i:j], vocab, vocab_s, device, model_type))
        i = j

    return batches, order


if __name__ == '__main__':
    pass