import argparse
import t5bottleneck.load_t5 as t5
from python_transformers import set_seed
import t5bottleneck.load_data as data
import torch
import torch.nn.functional as F
import t5bottleneck.load_flan_t5 as flan_t5
from nltk.translate.bleu_score import sentence_bleu


def text_from_latent_code(model, input_ids, start=None, args=None):
    # ----------------------
    past = model.get_hidden(input_ids)
    # attention_mask = input_ids.ne(model.config.pad_token_id).long()
    # input_embed = model.enc_embed_weight(input_ids)
    # encoding = model.t5_model.encoder(input_ids=None, inputs_embeds=input_embed, attention_mask=attention_mask)[0]
    # _, _, past = model.ae(encoding, attention_mask)
    # ----------------------
    context_tokens = model.tokenizer.encode('</s>') if start == None else model.tokenizer.encode(start)
    args.top_k = 0
    args.top_p = 1.0
    args.temperature = 1
    length = 40

    out = sample_sequence_conditional(
        model=model.t5_model,
        context=context_tokens,
        past=past,
        length=length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        decoder_tokenizer=model.tokenizer,
        args=args
    )
    # print(out)

    text_x1 = model.tokenizer.decode(out[0,:].tolist()) # , clean_up_tokenization_spaces=True
    text_x1 = text_x1.split()
    text_x1 = ' '.join(text_x1)

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


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None, cvae=False, args=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    i=0
    with torch.no_grad():
        while i<length:
            inputs = {'input_ids': generated, 'encoder_hidden_states': past}
            if args.model_type == 't5':
                sequence_output = model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                sequence_output = sequence_output * (model.model_dim ** -0.5)
            elif args.model_type == 'bart':
                sequence_output = model.model.decoder(**inputs)[0]  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                sequence_output = sequence_output * (768 ** -0.5)
            else:
                exit()

            outputs = model.lm_head(sequence_output)
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            # print(next_token.unsqueeze(0)[0,0].item())
            if next_token.unsqueeze(0)[0,0].item() == 1:
                    break

            i+=1

    return generated


if __name__ == '__main__':
    model_dict = {'model_path': '', 't5_model_name': "t5-base", 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 1000, 'set_seq_size': 40,
                  'latent_vec': 'pooling', 'latent_type':'T5_original', 'latent_vec_dec': 'linear',
                  'pretrain_model_path':'checkpoints/flanT5/flan_t5_small_dec_e/train'}

    data_dict = {'train_data_file': 'math_symbolic_dataset/both_tr.txt',
                 'test_data_file': 'math_symbolic_dataset/both_te.txt',
                 'overwrite_cache': True}

    # parameter illustration:
    # output_dir: output dir for saving model
    # num_train_epochs: training epoch
    # per_device_train_batch_size: batch_size

    training_dict = {'output_dir': './output', 'overwrite_output_dir': True, 'do_train': True, 'do_eval': False,
                     'do_predict': False, 'evaluate_during_training': False, 'per_device_train_batch_size': 1,
                     'per_device_eval_batch_size': 1, 'per_gpu_train_batch_size': None, 'per_gpu_eval_batch_size': None,
                     'gradient_accumulation_steps': 1, 'learning_rate': 5e-05, 'weight_decay': 0.0, 'adam_epsilon': 1e-08,
                     'max_grad_norm': 1.0, 'num_train_epochs': 10, 'max_steps': -1, 'warmup_steps': 0,
                     'logging_dir': 'runs/Oct30_17-24-56_192.168.1.104', 'logging_first_step': False, 'logging_steps': -1,
                     'save_steps': -1, 'save_total_limit': 1, 'no_cuda': False, 'seed': 42, 'fp16': False, 'fp16_opt_level': 'O1',
                     'local_rank': -1, 'tpu_num_cores': None, 'tpu_metrics_debug': False, 'debug': False,
                     'dataloader_drop_last': False, 'eval_steps': 1000, 'past_index': -1, 'project_name': 'test',
                     'reg_schedule_k': 0.0025, 'reg_schedule_b': 6.25, 'reg_constant_weight': None,
                     'use_recon_loss': False}

    model_args = argparse.Namespace(**model_dict)
    data_args = argparse.Namespace(**data_dict)
    training_args = argparse.Namespace(**training_dict)

    args = {}
    args.update(model_dict)
    args.update(data_dict)
    args.update(training_dict)
    args = argparse.Namespace(**args)

    # check here before training.
    args.load_pretrain_t5 = True
    args.device = 'cpu'
    args.beta = 1.0
    args.log_interval = 10
    set_seed(training_args.seed)

    # checking latent connection mode.
    if model_args.latent_type == 'T5_original':
        if model_args.t5_model_name in ['t5-base', 't5-small']:
            model = t5.load_t5_origin(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_origin(model_args, training_args)
        elif model_args.t5_model_name in ["google/flan-t5-base", "google/flan-t5-small"]:
            model = flan_t5.load_flan_t5_original(model_args, training_args) if args.load_pretrain_t5 else flan_t5.new_flan_t5_original(model_args, training_args)
        else:
            exit('Error: wrong model name (two options: t5-base or google/flan-t5-base)')

    elif model_args.latent_type == 'BART_original':
        model = t5.load_bart_origin(model_args, training_args) if args.load_pretrain_t5 else t5.new_bart_origin(model_args, training_args)

    elif model_args.latent_type == 'T5_vae':
        model = t5.load_t5_vae(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_vae(model_args, training_args)
    else:
        model = t5.load_t5_ae(model_args, training_args) if args.load_pretrain_t5 else t5.new_t5_ae(model_args, training_args)

    model.to(args.device)

    # generate test set conclusion.
    train_dataset, test_dataset = data.get_dataset(data_args, tokenizer=model.tokenizer, set_seq_size=model_args.set_seq_size)

    test_dataloader = data.get_dataloader(args, test_dataset)

    index = 0
    scores_sum_bleu = 0
    acc = 0

    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):

            input, label = inputs['input_ids'], inputs['label_ids']
            input_ids = input.to(args.device)
            output_ids = label.to(args.device)

            conclusion = model.tokenizer.decode(output_ids.tolist()[0][1:], skip_special_tokens=True)

            gold_con = conclusion
            pred_con = text_from_latent_code(model, input_ids, args=args)

            # generated_ids = model.t5_model.generate(
            #     input_ids = input_ids,
            #     attention_mask = input_ids != 1,
            #     max_length=40,
            #     num_beams=2,
            #     repetition_penalty=2.5,
            #     length_penalty=1.0,
            #     early_stopping=True
            # )
            # pred_con = [model.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids][0].strip()
            # ------------------------------------------------------------------------------------

            print(str(step)+'##'*10)
            premises = model.tokenizer.decode(input_ids.tolist()[0])
            print("premises: ", premises)
            print("gold con: ", gold_con)
            print("pred con: ", pred_con)

            # ---------------------------------------------- BLEU --------------------------------------------------
            references = [gold_con.split(' ')]
            candidates = pred_con.split(' ')
            bleu_scores = sentence_bleu(references, candidates, weights=(1, 0, 0, 0))

            index += 1
            scores_sum_bleu += bleu_scores
            acc += 1

    print("bleu: ", scores_sum_bleu/index)
    print("accuracy: ", acc/index)
