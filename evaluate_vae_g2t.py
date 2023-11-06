import argparse
import os
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from optimus_g2t.data_access.math_symbolic import MathInferenceCorpus, MathReconstructCorpus, get_batches, conv_sent_dict
from optimus_g2t.examples.big_ae.utils import frange_cycle_zero_linear
from optimus_g2t.examples.big_ae.modules import OptimusVAE, VGraphAE
from python_transformers.optimization import AdamW, get_linear_schedule_with_warmup
from optimus_g2t.pytorch_transformers import (AdamW, BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch
import torch.nn.functional as F
from train_vae_g2t import load_graph_optimus, process_dataset, load_optimus_tokenizer, tokenize_function
from text_autoencoders.vocab import Vocab

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}
import logging


def generation_optimus(model, tokenizer_decoder, inputs, args=None, token_level='subword'):
    # attention_mask=(inputs > 0).float()
    # outputs = model.encoder(inputs.long(), attention_mask, role_ids=None)
    # pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
    # latent_z, _ = model.connect(pooled_hidden_fea)
    # past = latent_z.squeeze(1)
    z, _, graph_rec_loss, _ = model.encoder(inputs.x.to(args.device), inputs.edge_index.to(args.device))

    # Access the node features and batch assignment from the batch object
    node_features = z  # Shape: (total_num_nodes, num_node_features)
    batch_assignment = inputs.batch  # Shape: (total_num_nodes,)

    # Get the unique batch values (individual graph identifiers) from the batch assignment
    unique_batches = batch_assignment.unique()
    individual_graphs = []

    # Split the node features based on the batch assignment to form individual graphs
    for batch_val in unique_batches:
        mask = batch_assignment == batch_val
        individual_graph = node_features[mask]
        individual_graphs.append(torch.mean(individual_graph, dim=0))

    pooled_hidden_fea = torch.stack(individual_graphs).to(args.device)

    past, _ = model.connect(pooled_hidden_fea)
    past = past.squeeze(1)

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
    if token_level == 'char':
        return text_x1
    else:
        text_x1 = text_x1.split()
        text_x1 = ' '.join(text_x1[1:])
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


def evaluate(model, batches, args=None, logger=None):
    # model.eval()
    # model_vgae.eval()
    final_loss = 0
    with torch.no_grad():
        for i in range(len(batches)):
            inputs, labels = batches[i][0], batches[i][1].T
            loss_rec, loss_kl, loss, loss_rec_graph = model.autoenc(inputs, labels.long())

            if (i + 1) % args.log_interval == 0:
                # print('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))
                # logger.info('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}, graph {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec, loss_rec_graph))

            final_loss += loss

    return final_loss/len(batches)


def train_func(args, model, train_batches=None, tokenizer_decoder=None, token_level=None):
    acc = 0
    scores_sum_bleu = 0
    with torch.no_grad():
        for index, idx in enumerate(range(len(train_batches))):

            model.args.beta = 0
            model.args.lamb = 0
            model.args.fb_mode = 0

            inputs, labels = train_batches[idx][0], train_batches[idx][1].T

            pred_con = generation_optimus(model, tokenizer_decoder, inputs, args, token_level=token_level)
            print('#########')
            if token_level == 'char':
                gold_con = tokenizer_decoder.decode(labels.tolist()[0][:-1], clean_up_tokenization_spaces=True)
            else:
                gold_con = tokenizer_decoder.decode(labels.tolist()[0][1:-1], clean_up_tokenization_spaces=True)

            print(gold_con)
            print(pred_con)
            if gold_con.strip() == pred_con.strip():
                acc += 1

            # BLEU score
            references = [gold_con.split(' ')]
            candidates = pred_con.split(' ')
            bleu_scores = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))

            index += 1
            scores_sum_bleu += bleu_scores

    print("bleu: ", scores_sum_bleu/index)
    print("acc: ", acc/index)


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

    # ----------------------------------------
    token_level = 'char'
    include_var = True
    graph_type = 'TransformerConv' # GAT, GCN, GraphSAGE, TransformerConv
    # ----------------------------------------

    tokenizer_encoder, tokenizer_decoder = load_optimus_tokenizer(logger, token_level=token_level)

    # ------------------------------------------------------------------------------------------------------------------------------

    print('reconstruct task')
    train = MathReconstructCorpus(args.train_corpus)
    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': tokenizer_decoder, 'token_level': token_level}

    # encoding text for Text decoder.
    train_sents = []
    for sent in tqdm(train):
        tr_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        train_sents.append(tr_temp)

    # encoding graph for Graph encoder.
    train_dataset = process_dataset(dataset_path=args.train_corpus)
    _train = train_dataset.map(tokenize_function, batched=False)
    test_dataset = process_dataset(dataset_path=args.test_corpus)
    _test = test_dataset.map(tokenize_function, batched=False)

    train_sents_s, valid_sents_s = [], []
    vocab_sent_s = []
    for i in _train:
        train_sents_s.append(i)
        vocab_sent_s.append([node.strip() for node in i['equation1']['node_list']])
    for i in _test:
        valid_sents_s.append(i)
        vocab_sent_s.append([node.strip() for node in i['equation1']['node_list']])

    # building Vocab for graph encoder.
    vocab_file = os.path.join(args.save_dir, 'vocab_node.txt')
    vocab = Vocab(vocab_file)
    print("size of vocab: ", vocab.size)

    # load VGAE.
    # ------------------------------------------------------------------------------------------------------------------------------
    print("loading VGAE")
    # vocab = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8, 'rand': 9, 'pad': 10}
    model_vgae = VGraphAE(device, input_dim=args.latent_size, hidden_dim=args.latent_size, num_layers=2, sym_dict=vocab, heads=8, include_var=include_var, graph_type=graph_type)
    model = load_graph_optimus(args, logger, model_encoder=model_vgae, tokenizer_decoder=tokenizer_decoder)
    model = model.to(device)

    train_par = {'data': train_sents, 'data_struct': train_sents_s, 'batch_size': args.batch_size, 'model': model, 'device': device, 'include_var': include_var}
    train_batches, _ = get_batches(**train_par)
    # ------------------------------------------------------------------------------------------------------------------------------

    train_func(args, model, train_batches=train_batches, tokenizer_decoder=tokenizer_decoder, token_level=token_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':1,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        'train_corpus':'./math_symbolic_dataset/recon/both_tr_char.txt',
        'test_corpus': './math_symbolic_dataset/recon/both_te_char.txt',
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
        'pretrain_model_path': './checkpoint-768',
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
        'save_dir': 'checkpoints/test',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)

