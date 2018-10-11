from torch.utils.data import Dataset
from enum import Enum
import os
import random
from scipy.io import wavfile
from typing import *
import torch
from torch import nn
import re


def _tokenize_syntax(linearized_syntax) -> List[str]:
    """
    :param linearized_syntax:
    :return: a list of tokens
    """
    tokens = re.split("[)(]", linearized_syntax)
    return tokens + ["(", ")"]


def syntax_token_type(root: str) -> Dict[str, int]:
    """
    :param root: data_dir
    :return: a set of tokens used in linearized syntax tree
    """
    token_dict = {}
    nth_token_type = 0
    stx_filenames = [os.path.join(root, pair_path, "syntax.txt") for pair_path in os.listdir(root)]
    for filename in stx_filenames:
        with open(filename) as f:
            stx = f.read()
            tokens = _tokenize_syntax(stx)
            for t in tokens:
                if t not in token_dict:
                    token_dict[t] = nth_token_type
                    nth_token_type += 1
    return token_dict


def make_syntax_embedding(token_dict: Dict, trainable) -> nn.Embedding:
    # TODO: maybe I can make the embedding trainable and see what happens? (PCA and visualize?)
    if not trainable:  # simply use one-hot
        emb = nn.Embedding(len(token_dict), len(token_dict))
        emb.from_pretrained(torch.eye(len(token_dict)), freeze=True)
        return emb
    else:
        raise NotImplemented("No trainable embedding yet. What's the hurry, pal?")


class Phase(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class SpeechSyntax(Dataset):
    def __init__(self, root: str, phase: Phase, syntax_vocabulary: Dict[str, int]):
        """
        :param root: data dir
        :param phase: train, validate, or test
        :param syntax_vocabulary: a dictionary that maps syntax tokens to their indices into the embedding
        """
        self.phase = phase
        self.syntax_vocabulary = syntax_vocabulary
        # seeded random generator used only when partitioning data
        pair_paths = [os.path.join(root, pair_path) for pair_path in os.listdir(root)]
        # shuffle deterministically: "Answer to the Ultimate Question of Life, the Universe, and Everything"
        random.Random(42).shuffle(pair_paths)
        # split data into 7: 2: 1
        if phase == Phase.TRAIN:
            self.pair_paths = pair_paths[: int(0.7 * len(pair_paths))]
        elif phase == Phase.VALIDATE:
            self.pair_paths = pair_paths[int(0.7 * len(pair_paths)): int(0.9 * len(pair_paths))]
        else:
            assert phase == Phase.TEST
            self.pair_paths = pair_paths[int(0.9 * len(pair_paths)):]

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, index) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        gets us
            (1) speech audio tensor(torch.FloatTensor) (dim = 1)
            (2) syntax indices(torch.LongTensor) (dim = 1)
        both have variable length
        :return: (speech, syntax) tuples, both as torch.Tensor
        """
        # TODO: Make sure if I need to turn speech into one-hot?
        # TODO: If not, is there anything else I need to do to process it?
        path = self.pair_paths[index]
        # handle speech
        _, speech = wavfile.read(os.path.join(path, "speech.wav"))  # speech is 1-D
        speech = torch.from_numpy(speech)
        # handle syntax
        with open(os.path.join(path, "syntax.txt")) as f:
            syntax = f.read()
        syntax_idx = torch.LongTensor([self.syntax_vocabulary[c] for c in syntax])
        return speech, syntax_idx


def speech_syntax_collate_fn(data: List[Tuple[torch.FloatTensor, torch.LongTensor]]):
    """
    The default collate_fn in DataSet can't deal with variable-length input
    In this function we'll pack it and make it palatable for RNN
    :param data: a list of (speech syntax) tuple
    :return: a batch of packed speech, a batch of packed syntax (syntax indices)
    """
    # TODO
    pass
