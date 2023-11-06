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
        p, c = line[:-1].split('&')
        example = {
            'premises': p,
            'conclusion': c
        }

        examples.append(example)

    return examples


def MathReconstructCorpus(path, type='content'):
    with open(path, 'r') as file:
        # Read the entire text
        data = file.readlines()
    examples = []
    for i, line in enumerate(data):
        p, p1 = line[:-1].split('&')
        if type == 'content':
            example = {
                'premises': p,
                'conclusion': p
            }
        elif type == 'struct':
            example = {
                'premises': p1,
                'conclusion': p
            }
        else:
            example = {
                'premises': line[:-1],
                'conclusion': p
            }

        examples.append(example)

    return examples


def conv_sent_dict(sent, emb_tokenizer, decode_tokenizer, token_level, include_var, type):
    if token_level == 'subword':
        p = emb_tokenizer.encode(sent['premises'])
        c = decode_tokenizer.encode(sent['conclusion'])
    elif token_level == "subword_add_latex_tokens":
        p = emb_tokenizer.tokenize(sent['premises'].replace("[SEP]", ""))
        c = decode_tokenizer.tokenize(sent['premises'].replace("[SEP]", ""))
        p = emb_tokenizer.convert_tokens_to_ids(p)
        c = decode_tokenizer.convert_tokens_to_ids(c)
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
        if not include_var and type == 'content_struct':
            pattern = r"Symbol\([^)]+\)"
            p = re.sub(pattern, "Symbol", sent['premises'])
            c = re.sub(pattern, "Symbol", sent['conclusion'])
            p, struct = p.split('&')
            struct = emb_tokenizer.tokenize(struct)
        else:
            p, c = sent['premises'], sent['conclusion']
            struct = None

        # char-level
        c_p, c_c = [], []
        for i, s in enumerate(p.split('[SEP]')):
            s = s.strip()
            if s in ['where', 'the', 'variable', 'is'] and token_level == 'char_for_latex_only':
                c_p += [s]
                continue

            c_p += [c for c in list(s) if c.strip() != '']
            if i == len(p.split('[SEP]')) - 1:
                pass
            else:
                c_p += ['[SEP]']

        if type == 'content_struct':
            c_p += struct

        for i, s in enumerate(c.split('[SEP]')):
            s = s.strip()
            if s in ['where', 'the', 'variable', 'is'] and token_level == 'char_for_latex_only':
                c_c += [s]
                continue

            c_c += [c for c in list(s) if c.strip() != '']
            if i == len(c.split('[SEP]')) - 1:
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


def get_batch_infer(x, device):
    bert_pad = x[0]['bert_pad']
    gpt_pad = x[0]['gpt_pad']
    input_ids_bert = pad_sequence([torch.tensor(f['def_codes'], dtype=torch.long) for f in x], batch_first=True, padding_value=bert_pad)
    input_ids_gpt = pad_sequence([torch.tensor(f['conclusion'], dtype=torch.long) for f in x], batch_first=True, padding_value=gpt_pad)

    return input_ids_bert.t().to(device), input_ids_gpt.t().to(device)


def get_batches(data, batch_size, device):
    order = range(len(data))
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_infer(data[i: j], device))
        i = j

    return batches, order


if __name__ == '__main__':
    pass