from allennlp.predictors.predictor import Predictor
import re
import networkx as nx
import string
from tqdm import tqdm


def remove_word(sentence, tree):
    words_to_remove = sentence.split(' ')
    for w in words_to_remove:
        tree = tree.replace(w, "")
    return tree


def remove_punctuation(input_string):
    # Create a translation table to remove punctuation characters
    translator = str.maketrans('', '', string.punctuation)

    # Use the translate method to remove the punctuation characters
    cleaned_string = input_string.translate(translator)

    return cleaned_string


def parse_tree_string_to_list(tree_string):
    stack = []
    current = []
    word = ''

    for char in tree_string:
        if char == '(':
            if word:
                current.append(word)
                word = ''
            stack.append(current)
            current = []
        elif char == ')':
            if word:
                current.append(word)
                word = ''
            temp = current
            current = stack.pop()
            current.append(temp)
        elif char == ' ':
            if word:
                current.append(word)
                word = ''
        else:
            word += char

    return current


def parse_tree_to_graph(parse_tree):
        # Create a directed graph
        graph = nx.DiGraph()

        def traverse(node, parent=None):
            # Extract the label of the current node
            label = node[0]

            # Add the node to the graph
            graph.add_node(label)

            if parent:
                # Add an edge from the parent to the current node
                graph.add_edge(parent, label)

            # If the node has children, traverse them recursively
            if len(node) > 1:
                for child in node[1:]:
                    traverse(child, label)

        # Start the traversal from the root of the parse tree
        traverse(parse_tree)

        return graph


if __name__ == '__main__':
    pass

    # file_path = './natural_language_dataset/explanations_te.txt'
    # # Open the file in read mode
    # with open(file_path, 'r') as file:
    #     # Read the entire content of the file
    #     lines = file.readlines()
    # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    #
    # with open('explanations_parse_te.txt', 'w') as file:
    #     for line in lines:
    #         sentence = line.split('&')[0]
    #         if '/' in sentence or ';' in sentence or ',' in sentence:
    #             pass
    #         else:
    #             sentence = remove_punctuation(sentence)
    #             tree = predictor.predict(sentence)["trees"]
    #             print('#####')
    #             print(sentence)
    #             print(tree)
    #             file.write(sentence + ' & ' + tree + ' \n')

    # Example to convert a natural language to a graph.
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    sentence = "flower needs sun and oxygen"
    # # remove punctuation
    sentence = remove_punctuation(sentence)
    print(sentence)
    tree = predictor.predict(sentence)["trees"]
    print(tree)
    # print('original sentence: ', sentence)
    # # convert to parse tree string
    # tree = predictor.predict(sentence)["trees"]
    # print('original parse tree: ', tree)
    # # remove words only keep syntax types/
    # syntactic_elements = re.findall(r'[A-Z]+|\(|\)', tree)
    # tree = " ".join(syntactic_elements)
    # print('parse tree string: ', tree)
    #
    # parse_tree_list = parse_tree_string_to_list(tree)[0]
    # print("parse tree list: ", parse_tree_list)
    #
    # # Convert the parse tree to a graph
    # graph = parse_tree_to_graph(parse_tree_list)
    #
    # # Print the nodes and edges of the graph
    # print("Nodes:", graph.nodes())
    # print("Edges:", graph.edges())




