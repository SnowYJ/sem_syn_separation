import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import Dataset
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from graph.VGraphAE import VGraphAE
from torch_geometric.transforms import Pad
from graph.utils import match_parentheses, pad_collate
import os
import re
from train_optimus_disentangle_graph import parse_tree_to_graph, parse_tree_string_to_list
from text_autoencoders.vocab import Vocab


class Experiment:
    def __init__(self, learning_rate, epochs, batch_size, max_length, save_path, input_dim=384, cons_list_sin = ['log', 'exp', 'cos', 'Integer', 'sin', 'Symbol'], cons_list_dou = ['Mul', 'Add', 'Pow']):
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.input_dim = input_dim
        self.cons_list_sin = cons_list_sin # operation within equation
        self.cons_list_dou = cons_list_dou # operation within equation

        # PROCESS DATA
        exp = 'recon_natural'
        # tr_data_path = "math_symbolic_dataset/recon_graph/both_tr.txt"
        # te_data_path = "math_symbolic_dataset/recon_graph/both_te.txt"
        tr_data_path = "natural_language_dataset/explanations_parse_tr.txt"
        te_data_path = "natural_language_dataset/explanations_parse_te.txt"

        self.train_dataset = self.process_dataset(dataset_path=tr_data_path)
        self.tokenized_train_datasets = self.train_dataset.map(self.tokenize_function, batched=False)

        self.test_dataset = self.process_dataset(dataset_path=te_data_path)
        self.tokenized_test_datasets = self.test_dataset.map(self.tokenize_function, batched=False)

        # LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # reading vocab.
        train_sents_s, valid_sents_s = [], []
        vocab_sents_s = []
        for i in self.tokenized_train_datasets:
            train_sents_s.append(i)
            vocab_sents_s.append([node.strip() for node in i['equation1']['node_list']])
        for i in self.tokenized_test_datasets:
            valid_sents_s.append(i)
            vocab_sents_s.append([node.strip() for node in i['equation1']['node_list']])

        if exp == 'recon_natural':
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            vocab_file = os.path.join(save_path, 'vocab_s.txt')
            if not os.path.isfile(vocab_file):
                Vocab.build(vocab_sents_s, vocab_file, 10000)
            vocab_s = Vocab(vocab_file)
        else:
            vocab_s = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8}

        dic = {'exp': exp}
        args = argparse.Namespace(**dic)

        # load Model
        self.model = VGraphAE(self.device, sym_dict=vocab_s, args=args)
        self.batch_size = batch_size
        print('model architecture')
        print(self.model)
        self.save_path = save_path

    def process_dataset(self, dataset_path):
        formatted_examples = []
        with open(dataset_path, 'r') as file:
            # Read the entire text
            data = file.readlines()

        for line in data:
            p, p1 = line[:-1].split('&')
            formatted_examples.append({"equation1": p1, "target": p1})

        dataset = Dataset.from_list(formatted_examples)

        return dataset

    def construct_graph(self, examples):
        # print(examples)
        device = self.device
        edge_index = [[], []]
        node_list = match_parentheses(examples)
        idx = 0
        idx_flag = 0
        # symbol in ['Pow', 'Symbol', "'L'commutative=True", 'Integer', '-1']
        for symbol in node_list[: -1]:
            if symbol in self.cons_list_sin: # ['log', 'exp', 'cos', 'Integer', 'sin', 'Symbol']
                edge_index[0].append(idx)
                edge_index[1].append(idx+1)

                idx = idx + 1

            elif symbol in self.cons_list_dou: # cons_list_dou = ['Mul', 'Add', 'Pow']
                edge_index[0].append(idx)
                edge_index[1].append(idx+1)

                idx_flag = idx
                idx = idx + 1

            else:
                edge_index[0].append(idx_flag)
                edge_index[1].append(idx+1)
        edge_index = torch.tensor(edge_index, dtype = torch.long).to(device)

        examples = {"node_list": node_list, "edge_index": edge_index}

        # examples = {"var_idx": var_idx}
        return examples

    def construct_natural_language_graph(self, examples):
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

    def tokenize_function(self, examples, is_symbol=False):
        if is_symbol:
            examples["equation1"] = self.construct_graph(examples["equation1"])
            examples["target"] = self.construct_graph(examples["target"])
        else:
            examples["equation1"] = self.construct_natural_language_graph(examples["equation1"])
            examples["target"] = self.construct_natural_language_graph(examples["target"])
        return examples

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        #predictions = np.argmax(logits, axis=-1)
        majority_class_preds = [1 for pred in logits]
        majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=logits, references=labels)
        return score

    def evaluate(self):
        sum_loss = 0
        _pad = Pad()
        test_loader = DataLoader(self.tokenized_test_datasets.with_format("torch"), batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)
        # self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                tot_loss = 0
                tot_loss_rec = 0
                tot_loss_kl = 0
                for idx in range(len(batch)):
                    equation1 = batch[idx]["equation1"]
                    loss, rec_loss, kl_loss = self.model(equation1)
                    tot_loss += loss
                    tot_loss_rec += rec_loss
                    tot_loss_kl += kl_loss

                sum_loss += tot_loss/len(batch)
                if i % 10 == 0:
                    print('batch {:4d}/{:4d}, loss: {:.4f}, loss rec: {:.4f}, loss kl: {:.4f}'.format(i+1, len(test_loader), tot_loss/len(batch), tot_loss_rec/len(batch), tot_loss_kl/len(batch)))

        return sum_loss/len(test_loader)

            # self.evaluation()

    def train_and_eval(self):
        device = self.device
        self.model.to(device)

        train_loader = DataLoader(self.tokenized_train_datasets.with_format("torch"), batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)

        print("Start training...")
        steps = 0
        min_loss = 10000
        for e in range(self.epochs):
            self.model.train()
            print('--'*40)
            for i, batch in enumerate(train_loader):
                steps += 1
                optim.zero_grad()
                tot_loss = 0
                tot_loss_rec = 0
                tot_loss_kl = 0

                # loss, rec_loss, kl_loss = self.model(batch[0], batch[1])

                for idx in range(len(batch)):
                    equation1 = batch[idx]["equation1"]
                    loss, rec_loss, kl_loss = self.model(equation1)
                    tot_loss += loss
                    tot_loss_rec += rec_loss
                    tot_loss_kl += kl_loss

                if i % 10 == 0:
                    print('epoch {:3d}, batch {:4d}/{:4d}, loss: {:.4f}, loss rec: {:.4f}, loss kl: {:.4f}'.format(e+1, i+1, len(train_loader), tot_loss/len(batch), tot_loss_rec/len(batch), tot_loss_kl/len(batch)))

                tot_loss.backward()
                optim.step()

            test_loss = self.evaluate()
            if test_loss < min_loss:
                min_loss = test_loss
                print('saving model, loss {:.4f}'.format(test_loss))
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'training.bin'))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
    #                     help="Which dataset to use")
    # parser.add_argument("--model", type=str, default="gcn", nargs="?",
    #                     help="Which model to use")
    # parser.add_argument("--batch_size", type=int, default=32, nargs="?",
    #                     help="Batch size.")
    # parser.add_argument("--max_length", type=int, default=128, nargs="?",
    #                     help="Input Max Length.")
    # parser.add_argument("--epochs", type=int, default=1, nargs="?",
    #                     help="Num epochs.")
    # parser.add_argument("--lr", type=float, default=3e-5, nargs="?",
    #                     help="Learning rate.")
    # parser.add_argument("--neg", type=int, default=1, nargs="?",
    #                     help="Max number of negative examples")
    #
    # args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    experiment = Experiment(
        input_dim=384,
        learning_rate=3e-5,
        batch_size=30,
        max_length=128,
        epochs=10,
        save_path="checkpoints/debug"
    )
    experiment.train_and_eval()


