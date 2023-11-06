from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import argparse
import os
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import torch.nn.functional as F
from optimus_separate_graph_gpt2.data_access.math_symbolic import MathInferenceCorpus, MathReconstructCorpus, get_batches, conv_sent_dict
from optimus_separate_graph_gpt2.examples.big_ae.modules import OptimusVAE, VGraphAE
from optimus_separate_graph_gpt2.pytorch_transformers import (AdamW, BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
from train_optimus_separate_graph_gpt2 import load_graph_optimus, load_optimus_tokenizer, process_dataset
import pickle
import numpy as np
import logging
from text_autoencoders.vocab import Vocab
import re

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}
from train_optimus_separate_graph_gpt2 import construct_natural_language_graph, construct_graph


def generation_optimus(model, tokenizer_decoder, past, args=None, token_level='subword'):

    context_tokens = tokenizer_decoder.encode('<BOS>')
    args.top_k = 0
    args.top_p = 1.0
    args.temperature = 1
    length = 40

    out = sample_sequence_conditional(
        model=model.decoder,
        context=context_tokens,
        past=past,
        length=length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        decoder_tokenizer=tokenizer_decoder
    )

    text_x1 = tokenizer_decoder.decode(out[0, :].tolist()) # , clean_up_tokenization_spaces=True
    if token_level == 'subword':
        text_x1 = text_x1.split()
        text_x1 = ' '.join(text_x1[1:])
        return text_x1
    elif token_level == 'char_add_latex_tokens_without_var':
        text_x1 = text_x1.split()
        text_x1 = ' '.join(text_x1).replace("<BOS>", "").strip()

        latex_token = ["\\frac", "\\sin", "\\cos", "\\log", "e^"]

        if any([s in text_x1 for s in latex_token]):
            pattern = '|'.join(re.escape(token) for token in latex_token)
            tokenized_parts = re.split(f'({pattern})', text_x1)
            result = []
            for part in tokenized_parts:
                if part in latex_token:
                    result.append(part)
                else:
                    result.extend(list(part))
            p = [item.strip() for item in result if item.strip() != '']
        else:
            p = [i.strip() for i in list(text_x1) if i.strip() != '']

        return ' '.join(p)
    else:
        text_x1 = text_x1.replace("<BOS>", "").strip()
        text_x1 = text_x1.replace("[SEP]", "").strip()
        return text_x1


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None, cvae=False):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    i=0
    with torch.no_grad():
        while i<length:
            # for _ in trange(length):
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            if next_token.unsqueeze(0)[0,0].item() == decoder_tokenizer.encode('<EOS>')[0]:
                break

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            i+=1

    return generated


def train_func(args, model, train_batches=None, tokenizer_decoder=None, token_level=None):
    model.eval()
    acc = 0
    index = 0
    scores_sum_bleu = 0
    scores_sum_cos, scores_sum_bleurt = 0, 0
    eval_latent_arr = []
    # bleurt model
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 model
    sentenceT5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    for i, idx in enumerate(range(len(train_batches))):

        model.args.beta = 0
        model.args.lamb = 0
        model.args.fb_mode = 0

        inputs_s, inputs_l, labels = train_batches[idx][0], train_batches[idx][1].T, train_batches[idx][2].T
        z = model.get_concat_latent(inputs_l, inputs_s)
        eval_latent_arr.append(z.tolist()[0])
        pred_con = generation_optimus(model, tokenizer_decoder, z, args=args, token_level=token_level)

        print('#########')
        if token_level == 'subword':
            gold_con = tokenizer_decoder.decode(labels.tolist()[0][1:-1], clean_up_tokenization_spaces=True)
        elif token_level == 'char_add_latex_tokens_without_var':
            gold_con = tokenizer_decoder.decode(labels.tolist()[0][1:-1], clean_up_tokenization_spaces=True)
            latex_token = ["\\frac", "\\sin", "\\cos", "\\log", "e^"]

            # tokenize for gold output.
            if any([s in gold_con for s in latex_token]):
                pattern = '|'.join(re.escape(token) for token in latex_token)
                tokenized_parts = re.split(f'({pattern})', gold_con)
                result = []
                for part in tokenized_parts:
                    if part in latex_token:
                        result.append(part)
                    else:
                        result.extend(list(part))
                p = [item.strip() for item in result if item.strip() != '']
            else:
                p = [i.strip() for i in list(gold_con) if i.strip() != '']

            gold_con = ' '.join(p)
        else:
            gold_con = tokenizer_decoder.decode(labels.tolist()[0][:-1], clean_up_tokenization_spaces=True)
            gold_con = gold_con.replace("<BOS>", "").strip()
            gold_con = gold_con.replace("[SEP]", "").strip()
            if "<EOS>" in gold_con:
                gold_con = gold_con.split("<EOS>")[0]

        print("gold: ", gold_con)
        print("pred: ", pred_con)
        if gold_con.strip() == pred_con.strip():
            acc += 1

        # BLEU score
        references = [gold_con.split(' ')]
        candidates = pred_con.split(' ')
        bleu_scores = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))

        references = [gold_con]
        candidates = [pred_con]
        with torch.no_grad():
            bleurt_scores = bleurt_model(**bleurt_tokenizer(references, candidates, return_tensors='pt'))[0].squeeze().item()

        # ------------------------------------------- SentenceT5 -----------------------------------------------
        sentences = [pred_con, gold_con]
        embeddings = sentenceT5_model.encode(sentences)
        embed1 = torch.FloatTensor(embeddings[0])
        embed2 = torch.FloatTensor(embeddings[1])
        cos_scores = torch.cosine_similarity(embed1, embed2, dim=0)

        index += 1
        scores_sum_bleu += bleu_scores
        scores_sum_cos += cos_scores
        scores_sum_bleurt += bleurt_scores

    print("bleu: ", scores_sum_bleu/index)
    print("acc: ", acc/index)
    print('bleurt: ', scores_sum_bleurt/index)
    print('cos: ', scores_sum_cos/index)
    with open(args.pretrain_model_path+'/eval_latent_arr.pkl', 'wb') as f:
        pickle.dump(np.array(eval_latent_arr), f)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, "train_log.log"), mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # ----------------------------------------
    token_level = 'char_add_latex_tokens_without_var'
    graph_type = 'TransformerConv' # GAT, GCN, GraphSAGE, TransformerConv
    exp = 'symbol' # symbol or natural
    fuse_way = 'tensor_fuse' # bi_direction , uni_direction , layer_disentangle, tensor_fuse
    type = 'content' # GPT2 input: content, content_struct
    include_var = False # if type is 'content_struct', include var or not for GPT2 decoder.
    include_var_graph = False # for graph encoder.
    # ----------------------------------------

    tokenizer_encoder, tokenizer_decoder = load_optimus_tokenizer(logger, token_level=token_level, type=type)

    # ------------------------------------------------------------------------------------------------------------------------------

    print('reconstruct task')
    train = MathReconstructCorpus(args.train_corpus)

    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': tokenizer_decoder, 'token_level': token_level, 'include_var': include_var, 'type': type}

    # encoding text for Text decoder.
    train_sents, valid_sents = [], []
    for sent in tqdm(train):
        # encoding
        tr_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        train_sents.append(tr_temp)

    def tokenize_function(examples, is_symbol=exp == 'symbol'):
        if is_symbol:
            examples["equation1"] = construct_graph(examples["equation1"])
            examples["target"] = construct_graph(examples["target"])
        else:
            examples["equation1"] = construct_natural_language_graph(examples["equation1"])
            examples["target"] = construct_natural_language_graph(examples["target"])
        return examples

    # encoding graph for Graph encoder.
    train_dataset = process_dataset(dataset_path=args.train_corpus)
    _train = train_dataset.map(tokenize_function, batched=False)

    train_sents_s, valid_sents_s = [], []
    vocab_sent_s = []
    for i in _train:
        train_sents_s.append(i)
        vocab_sent_s.append([node.strip() for node in i['equation1']['node_list']])

    # building Vocab for graph encoder.
    if include_var_graph:
        vocab_file = os.path.join(args.save_dir, 'vocab_node.txt')
        if not os.path.isfile(vocab_file):
            Vocab.build(vocab_sent_s, vocab_file, 10000)
        vocab_s = Vocab(vocab_file)
        print("size of vocab: ", vocab_s.size)
    else:
        # vocab_s = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8}
        vocab_s = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8, 'rand': 9, 'pad': 10}

    # load VGAE.
    # ------------------------------------------------------------------------------------------------------------------------------
    print("loading VGAE")
    include_var = False if exp == 'symbol' else True
    # vocab = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8, 'rand': 9, 'pad': 10}
    model_vgae = VGraphAE(device, input_dim=int(args.latent_size/2), hidden_dim=int(args.latent_size/2), num_layers=2, sym_dict=vocab_s, heads=8, include_var=include_var_graph, graph_type=graph_type)
    model = load_graph_optimus(args, logger, graph_encoder=model_vgae, tokenizer_decoder=tokenizer_decoder, tokenizer_encoder=tokenizer_encoder, fuse_way=fuse_way)
    model = model.to(device)

    train_par = {'data': train_sents, 'data_struct': train_sents_s, 'batch_size': args.batch_size, 'model': model, 'device': device, 'include_var': include_var_graph}
    train_batches, _ = get_batches(**train_par)
    # ------------------------------------------------------------------------------------------------------------------------------

    train_func(args, model, train_batches=train_batches, tokenizer_decoder=tokenizer_decoder, token_level=token_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':1,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        'train_corpus':'./natural_language_dataset/explanations_parse_te.txt',
        'test_corpus': None,
        'dec':'greedy',
        'decay_factor':0.1,
        'decay_patience':0,
        'dim_d':512,
        'dim_emb':512,
        'dim_h':1024,
        'dim_z':256,
        'latent_size': 768,
        'model_loss_func': 'beta_vae',
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-05,
        'device': 'cuda',
        'model': 'optimus',
        'latent_as_gpt_memory': True,
        'latent_as_gpt_emb': False,
        'use_pretrained_optimus': False,
        'pretrain_model_path': './checkpoints/checkpoint_sep_optimus_char',
        'inference_premises_com': True,
        'inference_premises_sep': False,
        'dim_target_kl': 1.0,
        'fb_mode': 0,
        'length_weighted_loss': False,
        'beta': 0.0,
        'disc_capacity':'0.0,5.0,25000.0,30.0',
        'dropout': 0.5,
        'epochs': 30,
        'eval_dis': False,
        'eval_interval':1,
        'exp':'exp1',
        'input_eval':None,
        'input_train':None,
        'lambda_adv':0,
        'lambda_kl':0,
        'lambda_p':0,
        'latent_spec':{'cont': 10, 'disc': [20, 2, 2, 3]},
        'lm':None,
        'lm_ckpt':None,
        'load_model':'',
        'local_rank':-1,
        'log_dir':None,
        'log_interval':1,
        'lr':0.0005,
        'max_len':20,
        'model_type':'beta',
        'nlayers':1,
        'no_cuda':False,
        'noise':[0.0, 0.0, 0.0, 0.0],
        'pretrain':False,
        'print_loss':False,
        'print_traversal':False,
        'pt_lm':'t5-small',
        'save_dir': './checkpoints/checkpoint_sep_optimus_char',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)