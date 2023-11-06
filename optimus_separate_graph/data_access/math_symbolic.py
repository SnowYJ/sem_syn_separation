# read math reasoning dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch
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


def conv_sent_dict(sent, emb_tokenizer, decode_tokenizer, token_level='char'):
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
                c_c += [s]
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


def get_batch_graph(x, model, include_var=False):
    output_data = []
    for i, graph in enumerate(x):
        edge, node = torch.tensor(graph['equation1']['edge_index']), graph['equation1']['node_list']
        node_embed = model.graph_encoder.generate_edge_emb(node) if not include_var else model.graph_encoder.generate_edge_emb_all(node)
        output_data.append(Data(x=node_embed, edge_index=edge))
    return Batch.from_data_list(output_data, follow_batch=['batch'])


def get_batch_infer(x, x_struct, model, device, include_var=False):
    gpt_pad = x[0]['gpt_pad']
    bert_pad = x[0]['bert_pad']
    input_ids_bert = pad_sequence([torch.tensor(f['def_codes'], dtype=torch.long) for f in x], batch_first=True, padding_value=bert_pad)
    input_ids_gpt = pad_sequence([torch.tensor(f['conclusion'], dtype=torch.long) for f in x], batch_first=True, padding_value=gpt_pad)
    input_ids_s = get_batch_graph(x_struct, model, include_var)

    return input_ids_s, input_ids_bert.t().to(device), input_ids_gpt.t().to(device)


def get_batches(data, data_struct, batch_size, model, device, include_var=False):
    order = range(len(data))
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_infer(data[i: j], data_struct[i:j], model, device, include_var=include_var))
        i = j

    return batches, order


if __name__ == '__main__':
    pass