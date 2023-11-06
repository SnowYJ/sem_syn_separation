from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from train_optimus_math_symbolic import load_optimus
from nltk.translate.bleu_score import sentence_bleu
import argparse
import os
from tqdm import tqdm
from optimus.data_access.math_symbolic import MathInferenceCorpus, MathReconstructCorpus, get_batches, conv_sent_dict
from optimus.pytorch_transformers import (BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import math
import logging
import pickle
import re
import matplotlib.pyplot as plt

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}


def generation_optimus(model, tokenizer_decoder, inputs, args=None, token_level='char'):
    attention_mask=(inputs > 0).float()
    outputs = model.encoder(inputs.long(), attention_mask, role_ids=None)
    pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
    latent_z, _ = model.connect(pooled_hidden_fea)
    past = latent_z.squeeze(1)

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
        return text_x1, past.tolist()[0]
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

        return ' '.join(p), past.tolist()[0]
    else:
        text_x1 = text_x1.replace("<BOS>", "").strip()
        text_x1 = ' '.join(list(text_x1.split('[SEP]')[0])).strip() + " where the variable is " + ' '.join(list(text_x1.split('[SEP]')[-1])).strip()
        text_x1 = text_x1.replace("[SEP]", "").strip()
        return text_x1, past.tolist()[0]


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


# --------------------------------------------------- Traversal --------------------------------------------------------
class Traverser:
    def __init__(self, dim):
        self.dim = dim

    def traverse_continuous_line(self, idx, size, loc=0, scale=1):
        samples = np.zeros(shape=(size, self.dim))
        if idx is not None:
            cdf_traversal = np.linspace(0.2, 0.8, size)
            cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale)
            # cont_traversal = stats.norm.ppf(cdf_traversal)
            for i in range(size):
                samples[i, idx] = cont_traversal[i]
        return samples.astype('f')

    def traverse_continuous_line_control(self, idx, size, loc=0, scale=1, v=0, direct='left'):
        samples = np.zeros(shape=(size, self.dim))
        if idx is not None:
            prob = stats.norm.cdf(v, loc=loc, scale=scale)
            if direct == 'left':
                cdf_traversal = np.linspace(0.000000001, prob, size)
            else:
                cdf_traversal = np.linspace(prob, 0.999999999, size)

            cont_traversal = stats.norm.ppf(cdf_traversal, loc=loc, scale=scale)

            for i in range(size):
                samples[i, idx] = cont_traversal[i]
        return samples.astype('f'), sum(cont_traversal)/len(cont_traversal)


def decode_optimus(z, latent_z=None, model=None, tokenizer_decoder=None):
    """
    z: latent output of optimus
    latent_z: latent output of optimus given input.
    """

    # input = torch.tensor([tokenizer_decoder.bos_token_id for _ in range(args.num_sent)]).view(args.num_sent, -1)

    if latent_z is not None:
        # find the conlumn index with nonzero value and then replace by
        input_latent = torch.cat([latent_z for _ in range(args.num_sent)], 0)
        column_index = np.nonzero(np.array(z))[1][0]
        input_latent[:, column_index] = torch.tensor(z)[:, column_index]
    else:
        input_latent = torch.tensor(z).view(z.shape[0], -1)

    inputs, sents = [], []

    from optimus.examples.big_ae.run_latent_generation import text_from_latent_code

    args.top_k = 0
    args.top_p = 1.0
    args.temperature = 1.0

    for i in input_latent:
        res = text_from_latent_code(i.view(1, -1), model, args, tokenizer_decoder)
        sents.append(res)

    return sents


def traverse(seed, tokenizer_encoder, tokenizer_decoder, model):
    dim_z = 32
    args.num_sent = 4
    seed = tokenizer_encoder.convert_tokens_to_ids(seed.split())
    encode_input = torch.tensor(seed).unsqueeze(0)
    attention_mask = (encode_input > 0).float()
    outputs = model.encoder(encode_input.long(), attention_mask)[1]
    latent_z, _, mean, logvar = model.connect_traversal(outputs)
    latent_z = latent_z.squeeze(1)
    print("Origin: ", decode_optimus(latent_z, model=model, tokenizer_decoder=tokenizer_decoder))

    for i in np.arange(dim_z, step=1):
        # randomly choose four value from normal distribution where the mean and variance from model.
        loc, scale = mean[i], math.sqrt(math.exp(logvar[i]))
        # loc, scale = 0, 1
        samples = Traverser(dim_z).traverse_continuous_line(idx=i, size=args.num_sent, loc=loc, scale=scale)
        res = decode_optimus(samples, latent_z=latent_z, model=model, tokenizer_decoder=tokenizer_decoder)

        for ix, r in enumerate(res):
            print('Dim {}, Sent {}: {}'.format(i, ix, r))


# ---------------------------------------------------- Interpolation ---------------------------------------------------

def preprocess(sentence, stop_words):
    return [w for w in sentence.lower().split() if w not in stop_words]


def word_mover_distance(sent1, sent2, model, stopword):
    sent1 = preprocess(sent1, stopword)
    sent2 = preprocess(sent2, stopword)
    distance = model.wmdistance(sent1, sent2)
    return distance


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, "train_log.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # ------------------------
    exp = 'recon'
    token_level = 'subword' # 'char_add_latex_tokens_without_var' subword , char , char_for_latex_only , subword_add_latex_tokens
    include_var = False
    type = 'content' # content, struct, content_struct
    # ------------------------

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    tokenizer_encoder, tokenizer_decoder, model = load_optimus(args, logger, token_level=token_level)
    model = model.to(device)

    # bleurt model
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 model
    sentenceT5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    if exp == 'inference':
        print('inference task')
        test = MathInferenceCorpus(args.test_corpus)
    else:
        print('reconstruct task')
        test = MathReconstructCorpus(args.test_corpus, type='content')

    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': tokenizer_decoder, 'token_level': token_level, 'include_var': include_var, 'type': type}

    train_sents, valid_sents = [], []

    for sent in tqdm(test):
        val_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        valid_sents.append(val_temp)

    val_par = {'data': valid_sents, 'batch_size': args.batch_size, 'device': device}
    valid_batches, _ = get_batches(**val_par)

    model.eval()
    batches = valid_batches
    index = 0
    acc = 0
    eval_latent_arr = []
    scores_sum_cos, scores_sum_bleurt, scores_sum_bleu = 0, 0, 0
    with torch.no_grad():
        for i in range(len(batches)):
            inputs, labels = batches[i][0].T, batches[i][1].T
            tokenize_sent = [i.replace("Ġ", '') for i in tokenizer_decoder.convert_ids_to_tokens(labels.tolist()[0])]
            attention_mask=(inputs > 0).float()
            outputs = model.encoder(inputs.long(), attention_mask, role_ids=None)
            pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
            latent_z, _ = model.connect(pooled_hidden_fea)
            outputs = model.decoder(input_ids=labels, past=latent_z.squeeze(1), labels=labels.long(), label_ignore=model.pad_token_id)
            # layer = 11
            for layer in range(12):
                layer = 11
                attention_output = outputs[-1][layer]
                mean_attention_output = torch.mean(attention_output, dim=1).squeeze(0)
                attention_matrix = mean_attention_output.detach().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(attention_matrix, aspect='auto')
                plt.title(f"Layer {layer+1}: Averaging Head Attention")
                x = ['z']+tokenize_sent
                y = tokenize_sent
                plt.xticks(range(len(x)), x)
                plt.yticks(range(len(y)), y)
                plt.colorbar()
                plt.savefig(f"layer{layer+1}_optimus.pdf", format="pdf", bbox_inches="tight")
                plt.show()

                exit()

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
                gold_con = ' '.join(list(gold_con.split('[SEP]')[0])).strip() + " where the variable is " + ' '.join(list(gold_con.split('[SEP]')[-1])).strip()
                gold_con = gold_con.replace("[SEP]", "").strip()

            pred_con, z = generation_optimus(model, tokenizer_decoder, inputs, args=args, token_level=token_level)
            eval_latent_arr.append(z)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':1,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        'test_corpus': './natural_language_dataset/attention_visualize_debug.txt',
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
        'pretrain_model_path': 'checkpoints/optimus_beta_0_latent_768_epoch_100_batch_64',
        'inference_premises_com': True,
        'inference_premises_sep': False,
        'dim_target_kl': 1.0,
        'fb_mode': 0,
        'length_weighted_loss': False,
        'beta': 1.0,
        'disc_capacity':'0.0,5.0,25000.0,30.0',
        'dropout': 0.5,
        'epochs': 20,
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
        'log_interval':20,
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
        'save_dir': 'checkpoints/optimus_beta_0_latent_768_epoch_100_batch_64',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)