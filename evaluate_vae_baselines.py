import argparse
import os
from sklearn.neighbors import NearestNeighbors

from text_autoencoders.vocab import Vocab
from text_autoencoders.model import *
from text_autoencoders.utils import *
from text_autoencoders.batchify import get_batches
from train_vae_baselines import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--output', metavar='FILE',
                    help='output file name (in checkpoint directory)')
parser.add_argument('--data', metavar='FILE',
                    help='path to data file')

parser.add_argument('--enc', default='mu', metavar='M',
                    choices=['mu', 'z'],
                    help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--max-len', type=int, default=35, metavar='N',
                    help='max sequence length')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate on data file')
parser.add_argument('--ppl', action='store_true',
                    help='compute ppl by importance sampling')
parser.add_argument('--reconstruct', action='store_true',
                    help='reconstruct data file')
parser.add_argument('--sample', action='store_true',
                    help='sample sentences from prior')
parser.add_argument('--arithmetic', action='store_true',
                    help='compute vector offset avg(b)-avg(a) and apply to c')
parser.add_argument('--interpolate', action='store_true',
                    help='interpolate between pairs of sentences')
parser.add_argument('--latent-nn', action='store_true',
                    help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--m', type=int, default=100, metavar='N',
                    help='num of samples for importance sampling estimate')
parser.add_argument('--n', type=int, default=5, metavar='N',
                    help='num of sentences to generate for sample/interpolate')
parser.add_argument('--k', type=float, default=1, metavar='R',
                    help='k * offset for vector arithmetic')

parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')

def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

def encode(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        if args.enc == 'mu':
            zi = mu
        else:
            zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+args.batch_size], device=device)
        outputs = model.generate(zi, args.max_len, args.dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += args.batch_size
    return strip_eos(sents)

def calc_ppl(sents, m):
    batches, _ = get_batches(sents, vocab, args.batch_size, device)
    total_nll = 0
    with torch.no_grad():
        for inputs, targets in batches:
            total_nll += model.nll_is(inputs, targets, m).sum().item()
    n_words = sum(len(s) + 1 for s in sents)    # include <eos>
    return total_nll / len(sents), np.exp(total_nll / n_words)


if __name__ == '__main__':
    # args = parser.parse_args()
    dic = {'checkpoint': './text', 'output': 'rec',
           'data': None, 'enc': 'mu', 'dec': 'greedy',
           'batch_size': 256, 'max_len': 35, 'evaluate': False,
           'ppl': False, 'reconstruct': True, 'sample': False, 'arithmetic': False,
           'interpolate': False, 'latent_nn': False, 'm': 100, 'n': 5, 'k': 1, 'seed': 1111,
           'no_cuda': False}
    args = argparse.Namespace(**dic)

    vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, 'model.pt'))

    # bleurt model
    bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    bleurt_model.eval()

    # sentenceT5 model
    sentenceT5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    if args.evaluate:
        sents = load_sent(args.data)
        batches, _ = get_batches(sents, vocab, args.batch_size, device)
        meters = evaluate(model, batches)
        print(' '.join(['{} {:.2f},'.format(k, meter.avg)
            for k, meter in meters.items()]))

    if args.ppl:
        sents = load_sent(args.data)
        nll, ppl = calc_ppl(sents, args.m)
        print('NLL {:.2f}, PPL {:.2f}'.format(nll, ppl))

    if args.sample:
        z = np.random.normal(size=(args.n, model.args.dim_z)).astype('f')
        sents = decode(z)
        write_sent(sents, os.path.join(args.checkpoint, args.output))

    if args.reconstruct:
        # sents = load_sent(args.data)
        # ----------------------------------------------------------------------------------
        # # Explanations
        # import load_data as data
        # train_sents, valid_sents = [], []
        # train_X = data._get_text_sequences_rec(args.train)
        # valid_X = data._get_text_sequences_rec(args.valid)
        #
        # for i, j in train_X:
        #     t = i.split()
        #     train_sents.append(t)
        #
        # for i, j in valid_X:
        #     t = i.split()
        #     valid_sents.append(t)
        # ----------------------------------------------------------------------------------
        # Definitions
        import load_definition as definition
        corpus = definition.DefinitionSemanticRoleCorpus('wordnet')
        tr, val, te = definition.load_sample(corpus)
        train_sents = definition.load_sent(tr[:100])
        sents = definition.load_sent(val[:100])
        # ----------------------------------------------------------------------------------
        z = encode(sents)
        sents_rec = decode(z)
        index, scores_sum_cos, scores_sum_bleu, scores_sum_bleurt = 0, 0, 0, 0
        for i, txt in enumerate(sents):
            gold = ' '.join(txt)
            pred = ' '.join(sents_rec[i])
            references = [gold]
            candidates = [pred]

            with torch.no_grad():
                bleurt_scores = bleurt_model(**bleurt_tokenizer(references, candidates, return_tensors='pt'))[0].squeeze().item()

            references = [gold.split(' ')]
            candidates = pred.split(' ')
            bleu_scores = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))

            sentences = [pred, gold]
            embeddings = sentenceT5_model.encode(sentences)
            embed1 = torch.FloatTensor(embeddings[0])
            embed2 = torch.FloatTensor(embeddings[1])
            cos_scores = torch.cosine_similarity(embed1, embed2, dim=0)

            print('##########')
            print('pred: ', pred)
            print('gold: ', gold)

            index += 1
            scores_sum_cos += cos_scores
            scores_sum_bleu += bleu_scores
            scores_sum_bleurt += bleurt_scores

        print("bleu: ", scores_sum_bleu / index)
        print("bleurt: ", scores_sum_bleurt / index)
        print("cosine: ", scores_sum_cos / index)

        write_z(z, os.path.join(args.checkpoint, args.output+'.z'))
        write_sent(sents_rec, os.path.join(args.checkpoint, args.output+'.rec'))
        write_sent(sents, os.path.join(args.checkpoint, 'gold.rec'))

    if args.arithmetic:
        fa, fb, fc = args.data.split(',')
        sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
        za, zb, zc = encode(sa), encode(sb), encode(sc)
        zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
        sd = decode(zd)
        write_sent(sd, os.path.join(args.checkpoint, args.output))

    if args.interpolate:
        f1, f2 = args.data.split(',')
        s1, s2 = load_sent(f1), load_sent(f2)
        z1, z2 = encode(s1), encode(s2)
        zi = [interpolate(z1_, z2_, args.n) for z1_, z2_ in zip(z1, z2)]
        zi = np.concatenate(zi, axis=0)
        si = decode(zi)
        si = list(zip(*[iter(si)]*(args.n)))
        write_doc(si, os.path.join(args.checkpoint, args.output))

    if args.latent_nn:
        sents = load_sent(args.data)
        z = encode(sents)
        with open(os.path.join(args.checkpoint, args.output), 'w') as f:
            nn = NearestNeighbors(n_neighbors=args.n).fit(z)
            dis, idx = nn.kneighbors(z[:args.m])
            for i in range(len(idx)):
                f.write(' '.join(sents[i]) + '\n')
                for j, d in zip(idx[i], dis[i]):
                    f.write(' '.join(sents[j]) + '\t%.2f\n' % d)
                f.write('\n')
