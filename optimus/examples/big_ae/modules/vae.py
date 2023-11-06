import math
import torch
import torch.nn as nn

from .utils import log_sum_exp
from numbers import Number

import pdb
from torch.autograd import Variable
import logging
logger = logging.getLogger(__name__)


class OptimusVAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder,  tokenizer_encoder, tokenizer_decoder, args): # 
        super(OptimusVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.args = args
        self.nz = args.latent_size

        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder

        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]

        # connector: from Bert hidden units to the latent space
        # self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)

        if args.exp == 'exp_infer':
            if args.inference_premises_com:
                # self.linear_represent = nn.Linear(768, 768, bias=True) # layer to learn representation of p1&p2.
                self.linear_relation = nn.Linear(args.latent_size, args.latent_size, bias=True) # layer to learn relation between p1&p2 and conclusion.
                # self.linear_relation = nn.Sequential(nn.Linear(args.latent_size, args.latent_size, bias=True), nn.Linear(args.latent_size, args.latent_size, bias=True))
            elif args.inference_premises_sep:
                self.linear_relation = nn.Linear(args.latent_size*2, args.latent_size, bias=True)
            else:
                exit('option: inference_premises_com or inference_premises_sep')

        # Standard Normal prior
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def connect_traversal(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        mean_list = mean.squeeze(0).tolist()
        logvar_list = logvar.squeeze(0).tolist()

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL, mean_list, logvar_list

    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        logvar.fill_(.0)
        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def autoenc(self, inputs, labels, input_roles, label_roles, args, role_label_ignore, dataset_size=0, infer=False, sep_infer_input=None):
        if args.model == 'optimus':
            if args.exp in ['exp1', 'exp4_train', 'exp4_gen', 'exp_infer']:
                w_loss_rec, w_loss_kl, w_loss = self(inputs=inputs.long(), labels=labels, dataset_size=dataset_size, infer=infer, sep_infer_input=sep_infer_input)
                w_loss_rec, w_loss_kl, w_loss = w_loss_rec.mean(), w_loss_kl.mean(), w_loss.mean()
                return (w_loss_rec, w_loss_kl, w_loss)

            elif args.exp == 'exp2':
                w_loss_rec, w_loss_kl, w_loss = self(inputs=inputs, labels=labels, role_label_ignore=role_label_ignore, dataset_size=dataset_size, infer=infer)
                w_loss_rec, w_loss_kl, w_loss = w_loss_rec.mean(), w_loss_kl.mean(), w_loss.mean()

                r_loss_rec, r_loss_kl, r_loss = self(inputs=None, labels=None, input_roles=input_roles, label_roles=label_roles, is_role=True, role_label_ignore=role_label_ignore, dataset_size=dataset_size, infer=infer)
                r_loss_rec, r_loss_kl, r_loss = r_loss_rec.mean(), r_loss_kl.mean(), r_loss.mean()
                return (w_loss_rec+r_loss_rec, w_loss_kl+r_loss_kl, w_loss+r_loss)

            elif args.exp == 'exp3':
                w_loss_rec, w_loss_kl, w_loss = self(inputs=inputs, labels=labels, input_roles=input_roles, label_roles=label_roles, is_role=False, role_label_ignore=role_label_ignore, dataset_size=dataset_size, infer=infer)
                w_loss_rec, w_loss_kl, w_loss = w_loss_rec.mean(), w_loss_kl.mean(), w_loss.mean()
                return (w_loss_rec, w_loss_kl, w_loss)

            else:
                exit('ERROR: optimus training failure in autenc() func')

        elif args.model == 'conditional_optimus':
            w_loss_rec, w_loss_kl, w_loss = self(inputs=inputs, labels=labels, cvae=True)
            w_loss_rec, w_loss_kl, w_loss = w_loss_rec.mean(), w_loss_kl.mean(), w_loss.mean()
            return (w_loss_rec, w_loss_kl, w_loss)

        else:
            exit('ERROR: optimus training failure in autenc() func')

    def forward(self, inputs, labels, input_roles=None, label_roles=None, is_role=False, role_label_ignore=None, cvae=False, dataset_size=0, infer=False, sep_infer_input=None):

        # token_type_ids = torch.zeros_like(inputs)
        #
        # for i, v in enumerate(inputs):
        #     index0, index1 = (v == 102).nonzero(as_tuple=True)[0]
        #     token_type_ids[i][index0:index1+1] = 1

        # only for exp2:
        if is_role:
            inputs, labels = input_roles.long(), label_roles.long()
            input_roles, label_roles = None, None

        attention_mask = (inputs > 0).float() if not is_role else (inputs<role_label_ignore).float()
        reconstrution_mask = (labels != 50257).float() if not is_role else (labels != role_label_ignore).float() # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)

        outputs = self.encoder(inputs, attention_mask, is_role=is_role, role_ids=input_roles)
        pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc) 768 by 64
        pooled_hidden_fea1 = self.encoder(inputs, attention_mask, is_role=is_role, role_ids=input_roles)[1] if sep_infer_input != None else None # used in inference task. separate inputs.

        if self.args.fb_mode == 0:
            if self.args.model_loss_func == 'beta_vae':
                # beta VAE loss func
                latent_z, loss_kl = self.connect(pooled_hidden_fea)
                # not use in reconstruction, only used in inference.
                if sep_infer_input != None:
                    latent_z1, loss_kl = self.connect(pooled_hidden_fea1)

                    # way 1
                    latent_z = torch.cat([latent_z,latent_z1],dim=2).squeeze(1)

                    # way 2
                    # latent_z *= latent_z1
                    # latent_z = latent_z.squeeze(1)

            elif self.args.model_loss_func == 'tc_vae':
                # beta TC-VAE loss func
                latent_z, ICMI, TC, DWKL = self.tcvae(pooled_hidden_fea, dataset_size)
                # latent_z = self.linear_relation(latent_z)
                if sep_infer_input != None:
                    latent_z1, loss_kl = self.connect(pooled_hidden_fea1)
                    latent_z = torch.cat([latent_z,latent_z1],dim=2).squeeze(1)
            else:
                latent_z = 0
                exit('ERROR: loss function')

            if infer:
                latent_z = self.linear_relation(latent_z)

            latent_z = latent_z.squeeze(1)

            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels.long(), label_ignore=50257,
                                   is_role=is_role, role_label_ignore=role_label_ignore, role_ids=label_roles)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
    
        elif self.args.fb_mode == 1:
            if self.args.model_loss_func == 'beta_vae':
                # Connect hidden feature to the latent space
                mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
                latent_z = self.reparameterize(mu, logvar, nsamples=1)
                loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                kl_mask = (loss_kl > self.args.dim_target_kl).float()
                loss_kl = (kl_mask * loss_kl).sum(dim=1)

                if sep_infer_input != None:
                    mu, logvar = self.encoder.linear(pooled_hidden_fea1).chunk(2, -1)
                    latent_z1 = self.reparameterize(mu, logvar, nsamples=1)
                    loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
                    kl_mask = (loss_kl > self.args.dim_target_kl).float()
                    loss_kl = (kl_mask * loss_kl).sum(dim=1)
                    # latent_z *= latent_z1
                    # latent_z = latent_z.squeeze(1)
                    latent_z = torch.cat([latent_z,latent_z1], dim=2).squeeze(1)

            elif self.args.model_loss_func == 'tc_vae':
                # beta TC-VAE loss func
                latent_z, ICMI, TC, DWKL = self.tcvae(pooled_hidden_fea, dataset_size)
            else:
                latent_z = 0
                exit('ERROR: loss function')

            if infer:
                latent_z = self.linear_relation(latent_z)

            latent_z = latent_z.squeeze(1)

            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels.long(), label_ignore=50257, is_role=is_role, role_label_ignore=role_label_ignore, role_ids=label_roles)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        elif self.args.fb_mode == 2:
            # Connect hidden feature to the latent space
            latent_z, loss_kl = self.connect_deterministic(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)

            # past = self.decoder.linear(latent_z)
            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels.long(), label_ignore=self.pad_token_id)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        else:
            # --------------------- AutoEncoder setup ---------------------
            latent_z, _ = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            # Decoding
            outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels.long(), label_ignore=50257,
                                   is_role=is_role, role_label_ignore=role_label_ignore, role_ids=label_roles)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss_kl = torch.tensor(0, device=loss_rec.device)
            
        # pdb.set_trace()
        if self.args.model_loss_func == 'beta_vae':
            if self.args.length_weighted_loss:
                loss = loss_rec / sent_length + self.args.beta * loss_kl
            else:
                loss = loss_rec + self.args.beta * loss_kl

            return loss_rec, loss_kl, loss

        elif self.args.model_loss_func == 'tc_vae':
            loss = loss_rec + self.args.beta * TC + ICMI + self.args.lamb*DWKL

            return loss_rec, TC, loss
        else:
            print('loss not found')
            exit()


    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def log_density(self, sample, mu, logvar):
        c = torch.Tensor([math.log(2 * math.pi)]).type_as(sample.data)
        inv_var = torch.exp(-logvar)
        tmp = sample - mu
        return -0.5 * (tmp*tmp * inv_var + logvar + c)

    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)

    def tcvae(self, bert_fea, dataset_size):
        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1) # batch, latent_size
        z = self.reparameterize(mean, logvar, 1) # batch, 1, latent

        # log q(z|x)
        logqz_condx = self.eval_inference_dist(z, (mean, logvar)).squeeze(1) # log pdf(z, N(mu, sig))
        logpz = self.eval_prior_dist(z).squeeze(1) # log pdf(z, N(0, 1))

        # minibatch weighted sampling code from TC-VAE pdf x with trained mu & sig
        _logqz = self.log_density(z, mean.unsqueeze(0), logvar.unsqueeze(0))

        batch_size = z.size(0)

        if True:
            logqz_prodmarginals = (self.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (self.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            # logqz_prodmarginals = self.logsumexp(_logqz, dim=1, keepdim=False).sum(1)
            # logqz = self.logsumexp(_logqz.sum(2), dim=1, keepdim=False)
        else:
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = self.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = self.logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        # ICMI: index code MI, TC: total correlation DWKL: Dimension-wise KL
        ICMI = logqz_condx - logqz
        TC = logqz - logqz_prodmarginals
        DWKL = logpz - logqz_prodmarginals

        return z, ICMI.mean(), TC.mean(), DWKL.mean()

    def encoder_sample(self, bert_fea, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        mu, logvar = mu.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)

    def decode(self, z, strategy, K=10):
        """generate samples from z given strategy
        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """
        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x
        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)

    def log_probability(self, x, z, role_ids):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        outputs = self.decoder(input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id, role_ids=role_ids)
        loss_rec = outputs[0]
        return -loss_rec

    def loss_iw(self, x0, x1, x2, x3, nsamples=50, ns=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        # encoding into bert features
        bert_fea = self.encoder(x0, role_ids=x2)[1]
        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        ##################
        # compute KL
        ##################
        # pdb.set_trace()
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        # mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        ll_tmp, rc_tmp = [], []
        for _ in range(int(nsamples / ns)):

            # (batch, nsamples, nz)
            z = self.reparameterize(mu, logvar, ns)
            # past = self.decoder.linear(z)
            past = z
            # [batch, nsamples]
            log_prior = self.eval_prior_dist(z) # log(pdf(z, N(0, 1)))
            log_gen = self.eval_cond_ll(x1, past, x3) # cross-entropy of generated example.
            log_infer = self.eval_inference_dist(z, (mu, logvar)) # log(pdf(z, N(mu, sig)))

            # pdb.set_trace()
            log_gen = log_gen.unsqueeze(0).contiguous().view(z.shape[0],-1)


            # pdb.set_trace()
            rc_tmp.append(log_gen)
            ll_tmp.append(log_gen + log_prior - log_infer)


            
        
        log_prob_iw = log_sum_exp(torch.cat(ll_tmp, dim=-1), dim=-1) - math.log(nsamples)
        log_gen_iw = torch.mean(torch.cat(rc_tmp, dim=-1), dim=-1)

        return log_prob_iw, log_gen_iw , KL

    def nll_iw(self, x0, x1, nsamples, ns=1):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x0, x1:  two different tokenization results of x, where x is the data tensor with shape (batch, *). 
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        #.      this problem is to be solved in order to speed up

        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]

            # Chunyuan:
            # encoding into bert features
            pooled_hidden_fea = self.encoder(x0)[1]

            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder_sample(pooled_hidden_fea, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x1, z)
            log_infer_ll = self.eval_inference_dist(z, param)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """
        # (k^2)
        # prior ~ N(0, 1)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen

    def eval_cond_ll(self, x, z, role_ids):
        """compute log p(x|z)
        """
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0]*z_shape[1], x_shape[-1]) 
            z = z.contiguous().view(x_shape[0]*z_shape[1], z_shape[-1]) 

        return self.log_probability(x, z, role_ids)

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z

    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur,
                std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std))
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()
            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if iter_ >= self.args.mh_burn_in and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0:
                samples.append(cur.unsqueeze(1))

        return torch.cat(samples, dim=1)

    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]
        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]
        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        mean, logvar = self.encoder.forward(x)

        return mean

    def eval_inference_dist(self, z, param):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)
        mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density



    def calc_mi(self, test_data_batch, args):
        # calc_mi_v3
        import math 
        from modules.utils import log_sum_exp

        mi = 0
        num_examples = 0

        mu_batch_list, logvar_batch_list = [], []
        neg_entropy = 0.
        for batch_data in test_data_batch:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            (batch_size, nz)
            mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            x_batch, nz = mu.size()

            #print(x_batch, end=' ')

            num_examples += x_batch

            # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

            neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
            mu_batch_list += [mu.cpu()]
            logvar_batch_list += [logvar.cpu()]

            pdb.set_trace()

        neg_entropy = neg_entropy / num_examples
        ##print()

        num_examples = 0
        log_qz = 0.
        for i in range(len(mu_batch_list)):
            ###############
            # get z_samples
            ###############
            mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            
            # [z_batch, 1, nz]

            z_samples = self.reparameterize(mu, logvar, 1)

            z_samples = z_samples.view(-1, 1, nz)
            num_examples += z_samples.size(0)

            ###############
            # compute density
            ###############
            # [1, x_batch, nz]
            #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
            indices = np.arange(len(mu_batch_list))
            mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
            logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
            x_batch, nz = mu.size()

            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
            var = logvar.exp()

            # (z_batch, x_batch, nz)
            dev = z_samples - mu

            # (z_batch, x_batch)
            log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

            # log q(z): aggregate posterior
            # [z_batch]
            log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

        log_qz /= num_examples
        mi = neg_entropy - log_qz

        return mi



    def calc_au(self, eval_dataloader, args, delta=0.01):
        """compute the number of active units
        """
        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)

        # (1, nz)
        mean_mean = means_sum / cnt

        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)

        # (nz)
        au_var = var_sum / (cnt - 1)

        return (au_var >= delta).sum().item(), au_var

