from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from t5bottleneck.load_t5 import _get_config, t5_AE
import torch


def _get_flan_t5_origin_requirements(model_args, training_args):
    name = model_args.t5_model_name # "google/flan-t5-base"
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    config = _get_config(model_args)
    return config, t5_model, tokenizer


def new_flan_t5_original(model_args, training_args):
    config, t5_model, tokenizer = _get_flan_t5_origin_requirements(model_args, training_args)
    return t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)


def load_flan_t5_original(model_args, training_args):
    config, t5_model, tokenizer = _get_flan_t5_origin_requirements(model_args, training_args)
    t5 = t5_AE(config, t5_model, None, model_args.set_seq_size, tokenizer, model_args, training_args)
    checkpoint = torch.load(model_args.pretrain_model_path, map_location=torch.device('cpu'))
    t5.load_state_dict(checkpoint)
    print('loading pretrained flan_t5 successful.')
    return t5


if __name__ == '__main__':
    model_dict = {'model_path': '', 't5_model_name': "google/flan-t5-base", 'model_type': 't5', 'config_name': None,
                  'tokenizer_name': None, 'cache_dir': None, 'ae_latent_size': 1000, 'set_seq_size': 70,
                  'latent_vec': 'sentenceT5',
                  'latent_vec_dec': 'linear',
                  'latent_type':'T5_ae',
                  'pretrain_model_path':'checkpoints/t5_base_original_loss_0.61/train'}

    data_dict = {'train_data_file': 'datasets/tr_data.csv', 'test_data_file': 'datasets/te_data.csv', 'overwrite_cache': False}

    # parameter illustration:
    # output_dir: output dir for saving model
    # num_train_epochs: training epoch
    # per_device_train_batch_size: batch_size

    training_dict = {'output_dir': './output', 'overwrite_output_dir': True, 'do_train': True, 'do_eval': False,
                     'do_predict': False, 'evaluate_during_training': False, 'per_device_train_batch_size': 10,
                     'per_device_eval_batch_size': 8, 'per_gpu_train_batch_size': None, 'per_gpu_eval_batch_size': None,
                     'gradient_accumulation_steps': 1, 'learning_rate': 5e-05, 'weight_decay': 0.0, 'adam_epsilon': 1e-08,
                     'max_grad_norm': 1.0, 'num_train_epochs': 10, 'max_steps': -1, 'warmup_steps': 0,
                     'logging_dir': 'runs/Oct30_17-24-56_192.168.1.104', 'logging_first_step': False, 'logging_steps': -1,
                     'save_steps': -1, 'save_total_limit': 1, 'no_cuda': False, 'seed': 42, 'fp16': False, 'fp16_opt_level': 'O1',
                     'local_rank': -1, 'tpu_num_cores': None, 'tpu_metrics_debug': False, 'debug': False,
                     'dataloader_drop_last': False, 'eval_steps': 1000, 'past_index': -1, 'project_name': 'test',
                     'reg_schedule_k': 0.0025, 'reg_schedule_b': 6.25, 'reg_constant_weight': None,
                     'use_recon_loss': False}

    # model_args = argparse.Namespace(**model_dict)
    # data_args = argparse.Namespace(**data_dict)
    # training_args = argparse.Namespace(**training_dict)
    #
    # config, t5_model, tokenizer = _get_flan_t5_origin_requirements(model_args, training_args)
    # print(config)

    # model = new_flan_t5_original(model_args, training_args)
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    #
    # # inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
    # # outputs = model.generate(**inputs)
    # # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))