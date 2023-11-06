from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import argparse
import os
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import torch.nn.functional as F
from optimus_separate_graph_syntax_constraint_gpt2.data_access.math_symbolic import MathInferenceCorpus, MathReconstructCorpus, get_batches, conv_sent_dict
from optimus_separate_graph_syntax_constraint_gpt2.examples.big_ae.modules import OptimusVAE, VGraphAE
from optimus_separate_graph_syntax_constraint_gpt2.pytorch_transformers import (AdamW, BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
from train_optimus_separate_graph_syntax_constraint_gpt2 import load_graph_optimus, load_optimus_tokenizer, process_dataset
import pickle
import numpy as np
import logging
from text_autoencoders.vocab import Vocab
import re
import matplotlib.pyplot as plt
from allennlp.predictors.predictor import Predictor
import nltk
from construct_constituency_graph import remove_word

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


# def ornstein_uhlenbeck_traversal(args, input_sent, factor_index, optimus, tokenizer_encoder, tokenizer_decoder, inn,
#                                  w1=-0.05, w2=0.5):
#     """
#     This is the traversal function based on ornstein uhlenbeck process.
#     z_{k, t+1} = - lambda * z_{k, t} + sigma * W_t, W_t ~ N(0, 1)
#     """
#     optimus.eval()
#     inn.eval()
#
#     input_vec = torch.LongTensor([101] + tokenizer_encoder.encode(input_sent) + [102]).view(1, -1)
#
#     attention_mask = (input_vec > 0).float()
#     pooled_hidden_fea = optimus.encoder(input_vec, attention_mask)[1]
#     latent_z, loss_kl, mu, sig = optimus.connect_traversal(pooled_hidden_fea)
#     latent_z = mu.squeeze(1).unsqueeze(2).unsqueeze(3)
#     z_ss, logdet = inn(latent_z)
#     z_ss = list(z_ss)
#     factor_size = args.factor_config[factor_index]
#
#     print('input: ', input_sent)
#     for i in range(args.num_sent):
#         if i == 0:
#             for j in range(len(z_ss)):
#                 z_ss[j] = z_ss[j].repeat(args.num_sent, 1, 1, 1)
#
#             samples = np.random.multivariate_normal(mean=np.zeros(factor_size), cov=np.eye(factor_size),
#                                                     size=args.num_sent)
#
#         tmp_z_ss = z_ss
#         tmp_z_ss[factor_index][i, :, 0, 0] = w1 * tmp_z_ss[factor_index][i, :, 0, 0] + w2 * torch.FloatTensor(samples[i])
#
#     latent_z = inn.reverse(tuple(tmp_z_ss)).squeeze(2).squeeze(2)
#     res = decode_optimus(args, latent_z, optimus, tokenizer_decoder)
#     for n, r in enumerate(res):
#         print("sent {} : {}".format(n, r))
#     print(' ')
def random_sample_around_point(initial_point, num_samples, stddev):
    # Generate random numbers from a normal distribution
    dimension = 384

    samples = np.random.normal(loc=initial_point, scale=stddev, size=(num_samples, dimension))
    distances = np.linalg.norm(samples-initial_point, axis=1)
    # set threshold distance
    return samples, distances


def traverse_func(args, model, train_batches=None, tokenizer_decoder=None, token_level=None, traverse_space='semantic'):
    model.eval()
    num_sents = 50
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    for i, idx in enumerate(range(len(train_batches))):
        model.args.beta = 0
        model.args.lamb = 0
        model.args.fb_mode = 0
        inputs_s, inputs_l, labels = train_batches[idx][0], train_batches[idx][1].T, train_batches[idx][2].T
        source = " ".join([i.replace("Ġ", '') for i in tokenizer_decoder.convert_ids_to_tokens(labels.tolist()[0])][1:-1])
        print("source: ", source)
        tree = predictor.predict(source)["trees"]
        tree = remove_word(source, tree)
        print("source tree: ", tree)

        if traverse_space == 'semantic':
            # semantic space
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l, inputs_s, output_mu_sig=True)
            z_s = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm, distances = random_sample_around_point(mean_lm.detach().numpy()[0], 20000, logvar_lm.mul(0.5).exp().detach().numpy()[0])
            min_dist, max_dist = min(distances), max(distances)
            dist_range_list = np.linspace(min_dist, max_dist, 10).tolist()
            for index in range(len(dist_range_list)):
                if index + 1 == len(dist_range_list):
                    break
                min_d, max_d = dist_range_list[index], dist_range_list[index+1]
                k = np.where(np.logical_and(distances<=max_d, distances>=min_d))
                latent_z_tmp = z_lm[k]
                z_lm_tmp = torch.tensor(latent_z_tmp)
                z_s_tmp = z_s.repeat(z_lm_tmp.shape[0], 1)
                latent_z = torch.cat((z_lm_tmp, z_s_tmp), dim=1)
                traverse_sents_list = []
                for j, z in enumerate(latent_z[:num_sents]):
                    pred_con = generation_optimus(model, tokenizer_decoder, z.unsqueeze(0).to(torch.float32), args=args, token_level=token_level)
                    print(str(j)+" : "+pred_con)
                    traverse_sents_list.append(pred_con)
                sum_dist = 0
                for pred_con in traverse_sents_list:
                    traverse_tree = predictor.predict(pred_con)["trees"]
                    traverse_tree = remove_word(pred_con, traverse_tree)
                    dist = nltk.edit_distance(tree, traverse_tree)
                    sum_dist += dist
                print("avg edit distance: ", sum_dist/len(traverse_sents_list))
        else:
            # syntax space
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l, inputs_s, output_mu_sig=True)
            z_lm = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)
            z_s, distances = random_sample_around_point(mean_s.detach().numpy()[0], 20000, logvar_s.mul(0.5).exp().detach().numpy()[0])
            min_dist, max_dist = min(distances), max(distances)
            dist_range_list = np.linspace(min_dist, max_dist, 10).tolist()
            for index in range(len(dist_range_list)):
                if index + 1 == len(dist_range_list):
                    break
                min_d, max_d = dist_range_list[index], dist_range_list[index+1]
                k = np.where(np.logical_and(distances<=max_d, distances>=min_d))
                latent_z_tmp = z_s[k]
                z_s_tmp = torch.tensor(latent_z_tmp)
                z_lm_tmp = z_lm.repeat(z_s_tmp.shape[0], 1)
                latent_z = torch.cat((z_lm_tmp, z_s_tmp), dim=1)
                traverse_sents_list = []
                for j, z in enumerate(latent_z[:num_sents]):
                    pred_con = generation_optimus(model, tokenizer_decoder, z.unsqueeze(0).to(torch.float32), args=args, token_level=token_level)
                    print(str(j)+" : "+pred_con)
                    traverse_sents_list.append(pred_con)
                # sum_dist = 0
                # for pred_con in traverse_sents_list:
                #     pass

            z_s = torch.tensor(z_s)
            z_lm = z_s.repeat(z_s.shape[0], 1)
            latent_z = torch.cat((z_lm, z_s), dim=1)

        exit()



def interpolate_func(args, model, train_batches=None, tokenizer_decoder=None, token_level=None, traverse_space='semantic'):
    model.eval()
    num_sents = 50
    for i, idx in enumerate(range(len(train_batches))):
        model.args.beta = 0
        model.args.lamb = 0
        model.args.fb_mode = 0
        if i+1 == len(train_batches):
            break
        inputs_s_0, inputs_l_0, labels_0 = train_batches[idx][0], train_batches[idx][1].T, train_batches[idx][2].T
        source_sent = ' '.join([i.replace("Ġ", '') for i in tokenizer_decoder.convert_ids_to_tokens(labels_0.tolist()[0])])

        inputs_s_1, inputs_l_1, labels_1 = train_batches[idx+1][0], train_batches[idx+1][1].T, train_batches[idx+1][2].T
        target_sent = ' '.join([i.replace("Ġ", '') for i in tokenizer_decoder.convert_ids_to_tokens(labels_1.tolist()[0])])

        print("source: ", source_sent)
        print("target: ", target_sent)

        if traverse_space == 'semantic':
            # source
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l_0, inputs_s_0, output_mu_sig=True)
            z_s_0 = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm_0 = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)
            # target
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l_1, inputs_s_1, output_mu_sig=True)
            z_s_1 = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm_1 = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)

            for j in range(num_sents):
                t = j/num_sents
                z_lm_new = (1-t) * z_lm_0 + t * z_lm_1
                latent_z = torch.cat((z_lm_new, z_s_0), dim=1)
                pred_con = generation_optimus(model, tokenizer_decoder, latent_z, args=args, token_level=token_level)
                print("sent "+str(j)+" : "+pred_con)

        elif traverse_space == 'syntax':
            # source
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l_0, inputs_s_0, output_mu_sig=True)
            z_s_0 = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm_0 = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)
            # target
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l_1, inputs_s_1, output_mu_sig=True)
            z_s_1 = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm_1 = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)

            for j in range(num_sents):
                print(j)
                t = j/num_sents
                z_s_new = (1-t) * z_s_0 + t * z_s_1
                latent_z = torch.cat((z_lm_0, z_s_new), dim=1)
                pred_con = generation_optimus(model, tokenizer_decoder, latent_z, args=args, token_level=token_level)
                print(pred_con)
        else:
            # interpolate both semantic and syntax
            # source
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l_0, inputs_s_0, output_mu_sig=True)
            z_s_0 = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm_0 = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)
            # target
            mean_lm, logvar_lm, mean_s, logvar_s = model.get_concat_latent(inputs_l_1, inputs_s_1, output_mu_sig=True)
            z_s_1 = model.reparameterize(mean_s, logvar_s, 1).squeeze(0).squeeze(1)
            z_lm_1 = model.reparameterize(mean_lm, logvar_lm, 1).squeeze(0).squeeze(1)

            for j in range(num_sents):
                print(j)
                t = j/num_sents
                z_s_new = (1-t) * z_s_0 + t * z_s_1
                z_lm_new = (1-t) * z_lm_0 + t * z_lm_1
                latent_z = torch.cat((z_lm_new, z_s_new), dim=1)
                pred_con = generation_optimus(model, tokenizer_decoder, latent_z, args=args, token_level=token_level)
                print(pred_con)


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
    token_level = 'subword' # 'char_add_latex_tokens_without_var'
    graph_type = 'TransformerConv' # GAT, GCN, GraphSAGE, TransformerConv
    exp = 'natural' # symbol or natural
    fuse_way = 'add_syntax_query'
    # add_syn_Q_sem_KV
    # add_syntax_query
    # fuse_syntax_query
    # fuse_syn_Q_sem_KV

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

    traverse_func(args, model, train_batches=train_batches, tokenizer_decoder=tokenizer_decoder, token_level=token_level, traverse_space='syntax')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':1,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        'train_corpus':'./natural_language_dataset/attention_visualize_debug.txt',
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
        'use_pretrained_optimus': True,
        'pretrain_model_path': './checkpoints/optimus_graphenc_transformerCONV_add_syn_Q_sem_KV(encode)_beta_0_latent_768_epoch_100_batch_64',
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
        'save_dir': './checkpoints/optimus_graphenc_transformerCONV_add_syn_Q_sem_KV(encode)_beta_0_latent_768_epoch_100_batch_64',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)