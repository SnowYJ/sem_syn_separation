import argparse
import time
import os
import random
import collections
import torch

from text_autoencoders.model import DAE, VAE, AAE
from text_autoencoders.vocab import Vocab
from text_autoencoders.meter import AverageMeter
from text_autoencoders.utils import set_seed, logging, load_sent
from text_autoencoders.batchify import get_batches


parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', required=True,
                    help='path to validation file')
parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')
# Architecture arguments
parser.add_argument('--vocab-size', type=int, default=10000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--dim_z', type=int, default=768, metavar='D',
                    help='dimension of latent variable z')
parser.add_argument('--dim_emb', type=int, default=512, metavar='D',
                    help='dimension of word embedding')
parser.add_argument('--dim_h', type=int, default=1024, metavar='D',
                    help='dimension of hidden state per layer')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='number of layers')
parser.add_argument('--dim_d', type=int, default=512, metavar='D',
                    help='dimension of hidden state in AAE discriminator')
# Model arguments
parser.add_argument('--model_type', default='dae', metavar='M',
                    choices=['dae', 'vae', 'aae'],
                    help='which model to learn')
parser.add_argument('--lambda_kl', type=float, default=0.5, metavar='R',
                    help='weight for kl term in VAE')
parser.add_argument('--lambda_adv', type=float, default=0, metavar='R',
                    help='weight for adversarial loss in AAE')
parser.add_argument('--lambda_p', type=float, default=0, metavar='R',
                    help='weight for L1 penalty on posterior log-variance')
parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,K',
                    help='word drop prob, blank prob, substitute prob'
                         'max word shuffle distance')
# Training arguments
parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
#parser.add_argument('--clip', type=float, default=0.25, metavar='NORM',
#                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
# Others
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')

def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets in batches:
            losses = model.autoenc(inputs, targets)
            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)


    # ---------------------------- Prepare data ----------------------------------------
    # train_sents = load_sent(args.train)
    # logging('# train sents {}, tokens {}'.format(len(train_sents), sum(len(s) for s in train_sents)), log_file)
    # valid_sents = load_sent(args.valid)
    # logging('# valid sents {}, tokens {}'.format(len(valid_sents), sum(len(s) for s in valid_sents)), log_file)

    # ----------------------------------------------------------------------------------
    # Explanations
    import t5bottleneck.load_data as data
    train_sents, valid_sents = [], []
    train_X = data._get_text_sequences(args.train)
    valid_X = data._get_text_sequences(args.valid)

    for i, j in train_X:
        t = i.split()
        train_sents.append(t)

    for i, j in valid_X:
        t = i.split()
        valid_sents.append(t)

    # ----------------------------------------------------------------------------------
    # # Definitions
    # import load_definition as definition
    # corpus = definition.DefinitionSemanticRoleCorpus('wordnet')
    # tr, val, te = definition.load_sample(corpus)
    # train_sents = definition.load_sent(tr[:100])
    # valid_sents = definition.load_sent(val[:100])
    # ----------------------------------------------------------------------------------

    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(train_sents, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model_type](vocab, args).to(device)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    train_batches, _ = get_batches(train_sents, vocab, args.batch_size, device)
    valid_batches, _ = get_batches(valid_sents, vocab, args.batch_size, device)

    best_val_loss = None
    for epoch in range(args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            inputs, targets = train_batches[idx]
            losses = model.autoenc(inputs, targets, is_train=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging(log_output, log_file)

        valid_meters = evaluate(model, valid_batches)
        logging('-' * 80, log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            log_output += ' {} {:.2f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            log_output += ' | saving model'
            ckpt = {'args': args, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            best_val_loss = valid_meters['loss'].avg
        logging(log_output, log_file)
    logging('Done training', log_file)

if __name__ == '__main__':
    # args = parser.parse_args()
    # args.noise = [float(x) for x in args.noise.split(',')]
    dic = {'train': './math_symbolic_dataset/recon_new/tr_all_len_23_char.txt', 'valid': './math_symbolic_dataset/recon_new/te_all_len_23_char.txt',
            'save_dir': 'checkpoints_debug',
            'load_model': '', 'vocab_size': 10000, 'dim_z': 768, 'dim_emb': 512, 'dim_h': 1024,
            'nlayers': 1, 'dim_d': 512, 'model_type': 'dae', 'lambda_kl': 0.5, 'lambda_adv': 0,
            'lambda_p': 0, 'noise': [0.0, 0.0, 0.0, 0.0], 'dropout': 0.5,
            'lr': 0.0005, 'epochs': 1, 'batch_size': 1, 'seed': 1111,
            'no_cuda': False, 'log_interval': 100}

    args = argparse.Namespace(**dic)
    main(args)
