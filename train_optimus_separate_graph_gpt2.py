import argparse
import os
import json
from tqdm import tqdm

from optimus_separate_graph_gpt2.data_access.math_symbolic import MathInferenceCorpus, MathReconstructCorpus, get_batches, conv_sent_dict
from optimus_separate_graph_gpt2.examples.big_ae.utils import frange_cycle_zero_linear
from optimus_separate_graph_gpt2.examples.big_ae.modules import OptimusVAE, VGraphAE
from python_transformers.optimization import AdamW, get_linear_schedule_with_warmup
from optimus_separate_graph_gpt2.pytorch_transformers import (AdamW, BertConfig, BertForLatentConnector, BertTokenizer, GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
import torch

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}
import logging
from datasets import Dataset
from graph.utils import match_parentheses
from text_autoencoders.vocab import Vocab
# from graph.VGraphAE import VGraphAE
import re
from train_optimus_disentangle_graph import parse_tree_to_graph, parse_tree_string_to_list


def process_dataset(dataset_path):
    formatted_examples = []
    with open(dataset_path, 'r') as file:
        # Read the entire text
        data = file.readlines()

    for line in data:
        p, p1 = line[:-1].split('&')
        formatted_examples.append({"equation1": p1, "target": p1})

    dataset = Dataset.from_list(formatted_examples)

    return dataset


def construct_graph(examples):
    cons_list_sin = ['log', 'exp', 'cos', 'Integer', 'sin', 'Symbol']
    cons_list_dou = ['Mul', 'Add', 'Pow']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_index = [[], []]
    node_list = match_parentheses(examples)
    idx = 0
    idx_flag = 0
    for symbol in node_list[: -1]:
        if symbol in cons_list_sin:
            edge_index[0].append(idx)
            edge_index[1].append(idx+1)
            idx = idx + 1
        elif symbol in cons_list_dou:
            edge_index[0].append(idx)
            edge_index[1].append(idx+1)
            idx_flag = idx
            idx = idx + 1
        else:
            edge_index[0].append(idx_flag)
            edge_index[1].append(idx+1)
    edge_index = torch.tensor(edge_index, dtype = torch.long).to(device)
    examples = {"node_list": node_list, "edge_index": edge_index}
    return examples


def construct_natural_language_graph(examples):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # remove words only keep syntax types/
    syntactic_elements = re.findall(r'[A-Z]+|\(|\)', examples)
    tree = " ".join(syntactic_elements)
    # print('parse tree string: ', tree)

    parse_tree_list = parse_tree_string_to_list(tree)[0]
    # print("parse tree list: ", parse_tree_list)

    # Convert the parse tree to a graph
    graph = parse_tree_to_graph(parse_tree_list)

    # Print the nodes and edges of the graph
    node_list = list(graph.nodes())
    edge_list = graph.edges()

    edge_index = [[], []]
    for tup in edge_list:
        edge_index[0].append(node_list.index(tup[0]))
        edge_index[1].append(node_list.index(tup[1]))

    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    examples = {"node_list": node_list, "edge_index": edge_index}

    return examples


def add_args(parser):
    # Data Locations
    # 'entailmentbankREC' without conclusion for reconstruction only.
    # 'entailmentbankINF' sentences pair (premises, conclusion).
    # 'entailmentbankCON' with conclusion for strategy 2.
    parser.add_argument('--corpus', default='debug', metavar='C', required=False,
                        choices=['debug', 'wordnet', 'wiktionary', 'wikipedia',
                                 'entailmentbankREC', 'entailmentbankINF', 'entailmentbankCON'],
                        help='corpus to be used')
    parser.add_argument("--lm", type=str, required=False, nargs="+",
                        help="location of txt file with text for LM pre-training")
    parser.add_argument('--lm_ckpt', type=str, required=False,
                        help="location of pretrained LM")
    parser.add_argument('--input_train', type=str, required=False,
                        help="location of train vectors for Input conditioning")
    parser.add_argument('--input_eval', type=str, required=False,
                        help="location of eval vectors for Input conditioning")
    parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                        help='directory to save checkpoints and outputs')
    parser.add_argument('--log-dir', metavar='DIR',
                        help='only used to copy log from localscratch')
    parser.add_argument("--w2v_weights", type=str, required=False,
                        help="path to pretrained embeddings to init")
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='DDP Local process rank.')


    # Data Settings
    parser.add_argument("--pretrain", action='store_true', help='pretrain LM flag')
    parser.add_argument('--load-model', default='', metavar='FILE', help='path to load checkpoint if specified')

    # Architecture arguments
    parser.add_argument('--dim_z', type=int, default=128, metavar='D', help='dimension of latent variable z')
    parser.add_argument('--dim_emb', type=int, default=512, metavar='D', help='dimension of word embedding')
    parser.add_argument('--dim_h', type=int, default=1024, metavar='D', help='dimension of hidden state per layer')
    parser.add_argument('--nlayers', type=int, default=1, metavar='N', help='number of layers')
    parser.add_argument('--dim_d', type=int, default=512, metavar='D', help='dim of hidden state in AAE discriminator')

    # Model arguments
    parser.add_argument('--pt_lm', default='t5-small', metavar='M',
                        choices=['t5-base', 't5-small', 'patrickvonplaten/t5-tiny-random'],
                        help='pre-trained emb LM')
    parser.add_argument('--model_type', default='dae', metavar='M',
                        choices=['beta', 'ann', 'dae', 'vae', 'aae', 'dm'],
                        help='which model to learn')
    parser.add_argument('--latent_spec', default='{"cont": 10,"disc": [20,2,2,3]}', type=json.loads)
    parser.add_argument('--cont_capacity', default="0.0,5.0,25000.0,30.0", type=str)
    parser.add_argument('--disc_capacity', default="0.0,5.0,25000.0,30.0", type=str)

    parser.add_argument('--eval_dis', action='store_true', help='evaluation')
    parser.add_argument('--print_traversal', action='store_true', help='print_traversal')
    parser.add_argument('--print_loss', action='store_true', help='print loss')
    parser.add_argument('--eval_interval', type=int, default=1, metavar='N', help='report eval')

    parser.add_argument('--lambda_kl', type=float, default=0, metavar='R',
                        help='weight for kl term in VAE')
    parser.add_argument('--lambda_adv', type=float, default=0, metavar='R',
                        help='weight for adversarial loss in AAE')
    parser.add_argument('--lambda_p', type=float, default=0, metavar='R',
                        help='weight for L1 penalty on posterior log-variance')
    parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,K',
                        help='word drop prob, blank prob, substitute prob'
                             'max word shuffle distance')
    # Training arguments
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP', help='dropout prob, 0 = no dropout')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--decay_patience', type=float, default=0, help='decay patience')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='batch size')

    # Others
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--max-len', type=int, default=20, metavar='N',
                        help='max sequence length')
    parser.add_argument('--dec', default='greedy', metavar='M',
                        choices=['greedy', 'sample'], help='decoding algorithm')

    parser.add_argument('--exp', default='exp3', metavar='M',
                        choices=['exp1', 'exp2', 'exp3', 'exp3_2', 'exp4_train', 'exp4_gen',
                                 'exp_infer'],
                        help='type of batch to retrieve')

    # Optimus
    parser.add_argument('--latent_size', type=int, default=32, metavar='N', help='latent size of optimus')
    parser.add_argument('--model', default='optimus', metavar='M', choices=['optimus', 'conditional_optimus', 'TransformerCVAE'],
                        help='latent size of optimus')
    parser.add_argument('--latent_as_gpt_memory', default=True, help='optimus latent injection 1')
    parser.add_argument('--latent_as_gpt_emb', default=False, help='optimus latent injection 2')

    parser.add_argument('--use_pretrained_optimus', action='store_true', help='whether use pre-trained optimus')
    parser.add_argument('--dim_target_kl', default=1.0, type=float, help='optimus calculating kl')
    parser.add_argument('--fb_mode', default=0, type=int, help='optimus calculating kl')
    parser.add_argument('--beta', default=2.0, type=int, help='optimus final kl weight')
    parser.add_argument('--length_weighted_loss', action='store_true')
    parser.add_argument('--model_loss_func', default='beta_vae', choices=['beta_vae', 'tc_vae'], help='type of loss function (beta vae and tcvae)')
    parser.add_argument('--gradient_accumulation_steps', default=3, type=int)

    parser.add_argument('--inference_premises_com', action='store_true', help='NLI: p1 and p2 is feed into model as a single sentence')
    parser.add_argument('--inference_premises_sep', action='store_true', help='NLI: p1 and p2 are feed into model separately')


def load_graph_optimus(args, logger, graph_encoder, tokenizer_decoder, tokenizer_encoder, fuse_way):
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES['bert']
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES['gpt2']
    pretrained_full_path = args.pretrain_model_path +'/checkpoint-full/training.bin'

    # if args.use_pretrained_optimus:
    #     path = args.pretrain_model_path
    #     pretrained_encoder_path = path + '/checkpoint-encoder'
    #     pretrained_decoder_path = path + '/checkpoint-decoder'
    #     pretrained_full_path = path +'/checkpoint-full/training.bin'
    #
    #     model_encoder = encoder_model_class.from_pretrained(pretrained_encoder_path, latent_size=int(args.latent_size/2))
    #     model_encoder.to(args.device)
    #
    #     model_decoder = decoder_model_class.from_pretrained(pretrained_decoder_path, latent_size=args.latent_size)
    #     model_decoder.to(args.device)
    #     model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    # else:
    encoder_config = encoder_config_class.from_pretrained('bert-base-cased')
    model_encoder = encoder_model_class.from_pretrained('bert-base-cased', cache_dir=None, config=encoder_config, latent_size=int(args.latent_size/2))

    decoder_config = decoder_config_class.from_pretrained('gpt2')
    model_decoder = decoder_model_class.from_pretrained('gpt2', cache_dir=None, config=decoder_config, latent_size=args.latent_size, latent_as_gpt_emb=False, latent_as_gpt_memory=True, fuse=fuse_way)
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    model_encoder.resize_token_embeddings(len(tokenizer_encoder))

    if logger: logger.info('Optimus: pretrained BERT & GPT2 are successfully loaded')

    model = OptimusVAE(model_encoder, graph_encoder, model_decoder, tokenizer_decoder, args)

    if args.use_pretrained_optimus:
        checkpoint = torch.load(pretrained_full_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Optimus: pretrained optimus is successfully loaded')

    return model


def load_optimus_tokenizer(logger, token_level, type):
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES['bert']
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES['gpt2']
    print('using optimus tokenizer!')
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained('gpt2', do_lower_case=False)
    # special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}

    if token_level == 'subword':
        special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    else:
        special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', 'sep_token': '[SEP]'}

    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'special tokens to GPT2')

    if type in ['content_struct', 'struct']:
        print('adding new latex token into gpt2')
        symbol_list = ['log', 'Mul', 'exp', 'Add', 'Symbol', 'Pow', 'cos', 'Integer', 'sin']
        num_added_toks = tokenizer_decoder.add_tokens(symbol_list)

    # print('We have added', num_added_toks, 'latex tokens to GPT2')
    if logger: logger.info('We have added %s tokens to GPT2', str(num_added_toks))

    if token_level in ['subword_add_latex_tokens', 'char_add_latex_tokens_without_var']:
        latex_token = ["\\frac", "\\sin", "\\cos", "\\log", "e^"]
        num = tokenizer_encoder.add_tokens(latex_token)
        num = tokenizer_decoder.add_tokens(latex_token)
        print('We have added', num, ' tokens to BERT and GPT2')

    return tokenizer_encoder, tokenizer_decoder


def get_optimizers(args, model, num_training_steps):
    """
        Setup the optimizer and the learning rate scheduler, modified for when training with a VAE with an input-decoder.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    return optimizer, scheduler


def evaluate(model, batches, args=None, logger=None):
    # model.eval()
    # model_vgae.eval()
    final_loss = 0
    with torch.no_grad():
        for i in range(len(batches)):
            # inputs, labels = batches[i][0], batches[i][1].T
            # loss_rec, loss_kl, loss, loss_rec_graph = model.autoenc(inputs, labels.long())
            inputs_s, inputs_l, labels = batches[i][0], batches[i][1].T, batches[i][2].T
            loss_rec, loss_kl, loss, loss_rec_graph = model.autoenc(inputs_l, inputs_s, labels.long())

            if (i + 1) % args.log_interval == 0:
                # print('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))
                # logger.info('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec))
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | test loss {:.4f} , kl {:.4f} , latent rec {:.4f}, graph {:.4f}'.format(1, i, len(batches), loss, loss_kl, loss_rec, loss_rec_graph))

            final_loss += loss

    return final_loss/len(batches)


def train_func(args, model, train_batches=None, test_batches=None, logger=None):
    n_iter = int(args.epochs) * len(train_batches)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta,  n_cycle=1, ratio_increase=0.25, ratio_zero=0.25)
    len_train_dataloader = len(train_batches)
    t_total = int(len_train_dataloader // args.gradient_accumulation_steps * args.epochs)
    optimizer, scheduler = get_optimizers(args, model=model, num_training_steps=t_total)
    min_loss = 100000
    model.zero_grad()

    print("**** start running checking training loss in train_log file ****")
    logger.info("***** Running training *****")
    logger.info("  Task = %s", args.exp)
    logger.info("  Latent Size = %s", args.latent_size)
    logger.info("  Encoder = %s", 'BERT')
    logger.info("  Decoder = %s", 'GPT2')
    logger.info("  Num Epochs = %s", args.epochs)
    logger.info("  KL annealing = start from %s to %s", 0, args.beta)
    logger.info("  batch size per device = %s", args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        print('----'*30)
        logger.info('----'*30)
        # indices = range(len(train_batches))
        for i, idx in enumerate(range(len(train_batches))):

            cur_beta = beta_t_list[i + epoch*len_train_dataloader]
            model.args.beta = cur_beta
            model.args.lamb = 0
            model.args.fb_mode = 0 if cur_beta == 0.0 else 1

            inputs_s, inputs_l, labels = train_batches[idx][0], train_batches[idx][1].T, train_batches[idx][2].T
            loss_rec, loss_kl, loss, loss_rec_graph = model.autoenc(inputs_l, inputs_s, labels.long())

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps
                loss_kl /= gradient_accumulation_steps
                loss_rec /= gradient_accumulation_steps

            if (i + 1) % args.log_interval == 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches | train loss {:.4f} , kl {:.4f} , latent rec {:.4f}, graph {:.4f}'.format(epoch + 1, i, len_train_dataloader, loss, loss_kl, loss_rec, loss_rec_graph))
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | train loss {:.4f} , kl {:.4f} , latent rec {:.4f}, graph {:.4f}'.format(epoch + 1, i, len_train_dataloader, loss, loss_kl, loss_rec, loss_rec_graph))

            loss.backward(retain_graph=True)
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm: 1.0
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

        test_loss = evaluate(model, test_batches, args=args, logger=logger)

        if min_loss > test_loss:
            min_loss = test_loss
            # print('| epoch {:3d} end | avg test loss {:.4f}| saving model!'.format(epoch + 1, test_loss))
            logger.info('| epoch {:3d} end | avg test loss {:.4f}| saving model!'.format(epoch + 1, test_loss))
            # model_encoder_to_save = model.module.encoder if hasattr(model, 'module') else model.encoder
            # model_decoder_to_save = model.module.decoder if hasattr(model, 'module') else model.decoder
            model_to_save = model.module if hasattr(model, 'module') else model

            # save encoder & decoder & full
            # output_encoder_dir = os.path.join(args.save_dir, 'checkpoint-encoder')
            # output_decoder_dir = os.path.join(args.save_dir, 'checkpoint-decoder')
            output_full_dir = os.path.join(args.save_dir, 'checkpoint-full')

            # if not os.path.exists(output_encoder_dir):
            #     os.makedirs(output_encoder_dir)
            # if not os.path.exists(output_decoder_dir):
            #     os.makedirs(output_decoder_dir)
            if not os.path.exists(output_full_dir):
                os.makedirs(output_full_dir)

            # model_encoder_to_save.save_pretrained(output_encoder_dir)
            # torch.save(model.encoder.state_dict(), os.path.join(output_encoder_dir, 'training.bin'))
            # model_decoder_to_save.save_pretrained(output_decoder_dir)

            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }
            torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
        else:
            # print('| epoch {:3d} end | avg test loss {:.4f}| not saving model!'.format(epoch + 1, test_loss))
            logger.info('| epoch {:3d} end | avg test loss {:.4f}| not saving model!'.format(epoch + 1, test_loss))


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
    token_level = 'char_add_latex_tokens_without_var' # subword, char, char_for_latex_only, subword_add_latex_tokens
    graph_type = 'TransformerConv' # GAT, GCN, GraphSAGE, TransformerConv
    exp = 'symbol' # symbol or natural
    fuse_way = 'layer_disentangle' # bi_direction, uni_direction, layer_disentangle, tensor_fuse
    type = 'content' # GPT2 input: content, content_struct, struct
    include_var = False # if type is 'content_struct', include var or not, for decoder.
    include_var_graph = False # use var or not for graph encoder.
    # ----------------------------------------

    tokenizer_encoder, tokenizer_decoder = load_optimus_tokenizer(logger, token_level=token_level, type=type)

    # ------------------------------------------------------------------------------------------------------------------------------

    print('reconstruct task')
    train = MathReconstructCorpus(args.train_corpus)
    test = MathReconstructCorpus(args.test_corpus)

    conv_sent_dict_param = {'emb_tokenizer': tokenizer_encoder, 'decode_tokenizer': tokenizer_decoder, 'token_level': token_level, 'include_var': include_var, 'type': type}

    # encoding text for Text decoder.
    train_sents, valid_sents = [], []
    for sent in tqdm(train):
        # encoding
        tr_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        train_sents.append(tr_temp)

    for sent in tqdm(test):
        val_temp = conv_sent_dict(sent, **conv_sent_dict_param)
        valid_sents.append(val_temp)

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
    if include_var_graph:
        # include var
        vocab_file = os.path.join(args.save_dir, 'vocab_node.txt')
        if not os.path.isfile(vocab_file):
            Vocab.build(vocab_sent_s, vocab_file, 10000)
        vocab_s = Vocab(vocab_file)
        print("size of vocab: ", vocab_s.size)
    else:
        # don't include var
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

    val_par = {'data': valid_sents, 'data_struct': valid_sents_s,'batch_size': args.batch_size, 'model': model, 'device': device, 'include_var': include_var_graph}
    valid_batches, _ = get_batches(**val_par)
    # ------------------------------------------------------------------------------------------------------------------------------

    train_func(args, model, train_batches=train_batches, test_batches=valid_batches, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dic = {
        'batch_size':1,
        'cont_capacity':'0.0,5.0,25000.0,30.0',
        # 'train_corpus':'./math_symbolic_dataset/recon/both_tr_char.txt',
        # 'test_corpus': './math_symbolic_dataset/recon/both_te_char.txt',
        'train_corpus':'./math_symbolic_dataset/recon/tr_all_len_23_char.txt',
        'test_corpus': './math_symbolic_dataset/recon/te_all_len_23_char.txt',
        # 'train_corpus':'./natural_language_dataset/explanations_parse_tr.txt',
        # 'test_corpus': './natural_language_dataset/explanations_parse_te.txt',
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
        'save_dir': 'checkpoints/debug',
        'seed':0,
        'w2v_weights':None
    }
    args = argparse.Namespace(**dic)
    main(args)