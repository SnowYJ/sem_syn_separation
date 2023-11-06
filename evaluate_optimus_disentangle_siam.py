from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from optimus_disentangle_siam.data_access.math_symbolic import MathInferenceCorpus, get_batches, conv_sent_dict, conv_sent_dict_s, MathReconstructCorpus
from optimus_disentangle_siam.examples.big_ae.modules import OptimusVAE
from optimus_disentangle_siam.pytorch_transformers import (BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
from text_autoencoders.vocab import Vocab
import re
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}
import logging
import pickle
import numpy as np


def generation_optimus(model, tokenizer_decoder, inputs, inputs_s, args=None, token_level=None):
    past = model.get_concat_latent(inputs, inputs_s)

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


def load_optimus(args, logger, tokenizer_encoder, tokenizer_decoder, vocab):
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES['bert']
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES['gpt2']

    if args.use_pretrained_optimus:
        path = args.pretrain_model_path
        pretrained_encoder_path = path + '/checkpoint-encoder'
        pretrained_decoder_path = path + '/checkpoint-decoder'
        pretrained_full_path = path +'/checkpoint-full/training.bin'

        model_encoder = encoder_model_class.from_pretrained(pretrained_encoder_path, latent_size=args.latent_size)
        model_encoder.to(args.device)
        model_decoder = decoder_model_class.from_pretrained(pretrained_decoder_path, latent_size=args.latent_size)
        model_decoder.to(args.device)
        model_decoder.resize_token_embeddings(len(tokenizer_decoder))
        model_encoder.resize_token_embeddings(len(tokenizer_encoder))
    else:
        encoder_config = encoder_config_class.from_pretrained('bert-base-cased')
        model_encoder = encoder_model_class.from_pretrained('bert-base-cased', cache_dir=None, config=encoder_config, latent_size=args.latent_size) # ,from_tf=bool('.ckpt' in 'bert-base-cased')

        decoder_config = decoder_config_class.from_pretrained('gpt2')
        model_decoder = decoder_model_class.from_pretrained('gpt2', cache_dir=None, config=decoder_config, latent_size=args.latent_size, latent_as_gpt_emb=False, latent_as_gpt_memory=True) # from_tf=bool('.ckpt' in 'gpt2')

        model_decoder.resize_token_embeddings(len(tokenizer_decoder))
        model_encoder.resize_token_embeddings(len(tokenizer_encoder))

    if logger: logger.info('Optimus: pretrained BERT & GPT2 are successfully loaded')

    model = OptimusVAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args, vocab)

    if args.use_pretrained_optimus:
        checkpoint = torch.load(pretrained_full_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Optimus: pretrained optimus is successfully loaded')

    return model


def load_optimus_tokenizer(logger, token_level):
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES['bert']
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES['gpt2']

    print('using optimus tokenizer!')
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained('gpt2', do_lower_case=False)
    if token_level == 'subword':
        special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    else:
        special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', 'sep_token': '[SEP]'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    if logger: logger.info('We have added %s tokens to GPT2', str(num_added_toks))

    if token_level in ['subword_add_latex_tokens', 'char_add_latex_tokens_without_var']:
        latex_token = ["\\frac", "\\sin", "\\cos", "\\log", "e^"]
        num = tokenizer_encoder.add_tokens(latex_token)
        num = tokenizer_decoder.add_tokens(latex_token)
        print('We have added', num, ' tokens to BERT and GPT2')

    return tokenizer_encoder, tokenizer_decoder


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, "train_log.log"), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # bleurt model
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 model
    sentenceT5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # ------------------------------------------------------------------------------------------------------------------
    exp = 'recon_symbol'
    include_var = False
    token_level = 'char_add_latex_tokens_without_var'
    # ------------------------------------------------------------------------------------------------------------------
    tokenizer_encoder, tokenizer_decoder = load_optimus_tokenizer(logger, token_level=token_level)

    if exp == 'inference':
        print('inference task')
        train = MathInferenceCorpus(args.train_corpus)
        test = MathInferenceCorpus(args.test_corpus)
    elif exp in ['recon_symbol', 'recon_natural']:
        print('reconstruction task')
        train = MathReconstructCorpus(args.train_corpus)
        test = MathReconstructCorpus(args.test_corpus)
    else:
        train, test = None, None

    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': tokenizer_decoder,
                            'is_symbol': exp!='recon_natural', 'include_var': include_var, 'token_level': token_level}

    bad_chars = ['\\', '{', '}', '(', ')']
    train_sents, valid_sents = [], []

    # Content (BoW)
    train_sents_bow, valid_sents_bow = [], []
    vocab_sents = []

    # Structure
    valid_sents_s = []
    for sent in tqdm(test):
        val_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        val_temp_s = conv_sent_dict_s(sent, **conv_sent_dict_param)

        valid_sents.append(val_temp)
        valid_sents_s.append(val_temp_s)
        # ----------------------
        # Content
        p, c = sent['premises'], sent['conclusion']
        if token_level == 'char_add_latex_tokens_without_var':
            p = p.split('[SEP]')[0].strip()
            c = c.split('[SEP]')[0].strip()
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

            vocab_sents.append(p)
            vocab_sents.append(c)
            valid_sents_bow.append({'premise': p, 'conclusion': c})
        else:
            for i in bad_chars:
                p = p.replace(i, ' ')
                c = c.replace(i, ' ')

            vocab_sents.append(p.split(' '))
            vocab_sents.append(c.split(' '))
            valid_sents_bow.append({'premise': [x for x in p.split(' ') if x != ''], 'conclusion': [x for x in c.split(' ') if x != '']})
        # ----------------------

    # building Vocab for BoW model.
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(vocab_sents, vocab_file, 10000)
    vocab = Vocab(vocab_file)
    print("size of vocab: ", vocab.size)

    val_par = {'data': valid_sents, 'data_bow': valid_sents_bow, 'data_struct': valid_sents_s,'batch_size': args.batch_size, 'vocab': vocab, 'device': device}
    valid_batches, _ = get_batches(**val_par)

    model = load_optimus(args, logger, tokenizer_encoder, tokenizer_decoder, vocab)
    model = model.to(device)

    model.eval()
    batches = valid_batches
    index = 0
    acc = 0
    scores_sum_cos, scores_sum_bleurt, scores_sum_bleu = 0, 0, 0
    eval_latent_arr = []
    with torch.no_grad():
        for i in range(len(batches)):
            inputs, labels = batches[i][0].T, batches[i][1].T
            inputs_s, labels_s = batches[i][4].T, batches[i][5].T

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

            # gold_con = tokenizer_decoder.decode(labels.tolist()[0][1:-1], clean_up_tokenization_spaces=True)
            pred_con, z = generation_optimus(model, tokenizer_decoder, inputs, inputs_s, args=args, token_level=token_level)
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
        'train_corpus':'./math_symbolic_dataset/recon_new/EVAL_new_char.txt',
        'test_corpus': './math_symbolic_dataset/recon_new/EVAL_new_char.txt',
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
        'use_pretrained_optimus': True,
        'pretrain_model_path': 'checkpoints_on_new_dataset/',
        'inference_premises_com': True,
        'inference_premises_sep': False,
        'dim_target_kl': 1.0,
        'fb_mode': 0,
        'length_weighted_loss': False,
        'beta': 0.0,
        'disc_capacity':'0.0,5.0,25000.0,30.0',
        'dropout': 0.5,
        'epochs': 50,
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
        'save_dir': 'checkpoints_on_new_dataset/test',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)