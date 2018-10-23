from torch.utils.data import Dataset
from enum import Enum
import os
import random
from scipy.io import wavfile
from typing import *
import torch
from torch import nn
from torch import utils
import re
import sys
import numpy as np
import librosa
from collections import defaultdict

"""
design: 
    single out blacklisted sentences(^XP^YP, ":") in syntax_token_type
    throw them away in Dataset.__init__
    in syntax_token_type, turn ^XP into XP since it's totally okay to do
"""


def _tokenize_syntax(linearized_syntax: str) -> List[str]:
    """
    :param linearized_syntax:
    :return: a list of tokens
    """
    tokens = re.split("[\[\]]", linearized_syntax)
    tokens.remove("")  # cause by things like [XX[]]
    return tokens + ["[", "]"]


def syntax_token_type(root: str) -> Tuple[Dict[str, int], Set[str]]:
    """
    :param root: data_dir
    :return: token_dict a dict of tokens used to lookup embedding
    :return: black_list a set of observations excluded from dataset
    REFERENCE: https://catalog.ldc.upenn.edu/docs/LDC99T42/
    """
    # CAVEAT: use it with unlexicalized syntax only
    token_set = defaultdict(int)
    blacklist = []
    stx_filenames = [os.path.join(root, pair_path, "syntax.txt") for pair_path in os.listdir(root)]
    for filename in stx_filenames:
        with open(filename) as f:
            stx = f.read()
            tokens = _tokenize_syntax(stx)
            bad_instance = False
            for i in range(len(tokens)):  # sanitize first
                t = tokens[i]
                if t == ":":  # totally unexplainable. probably just a typo
                    print("encountering a weird semicolon at {}, blacklisted".format(filename), file=sys.stderr)
                    blacklist.append(os.path.dirname(filename))
                    bad_instance = True
                elif re.match("^\^[A-Z]+\$?$", t):  # like ^VP, simply correct it
                    tokens[i] = t[1:]
                elif re.match("^\^[A-Z]+\$?\^[A-Z]+\$?$", t):  # like ^NNP^POS, which breaks the tree
                    print("encountering type II typo at {}, blacklisted".format(filename), file=sys.stderr)
                    blacklist.append(os.path.dirname(filename))
                    bad_instance = True
                elif "|" in t:  # the ambiguous case like "TO|IN", break tie arbitrarily
                    print("encountering ambiguous tag at {}, break-tied".format(filename), file=sys.stderr)
                    tokens[i] = t.split("|")[0]
            if bad_instance:  # avoid adding to token set
                continue
            for t in tokens:
                token_set[t] += 1
    # print("Syntax Token Count: ")
    # print(token_set)

    # TODO: According to the reference(https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid2.pdf),
    # TODO: the "^" on POS tag is to deal with the case where a word has wrong orthography
    # TODO: due to homophone but given correct POS tag anyway
    # TODO: I think it's totally okay to collapse things like "^VP" and "VP"
    token_set = list(token_set.keys())
    token_set.remove("")  # caused by splitting things like [S[]]

    tokens = sorted(token_set)  # make sure that the mapping of embedding is deterministic
    token_dict = {}
    for idx, t in enumerate(tokens):
        token_dict[t] = idx
    return token_dict, set(blacklist)


def make_syntax_embedding(syntax_vocabulary: Dict, trainable: bool) -> nn.Embedding:
    # TODO: maybe I can make the embedding trainable and see what happens? (PCA and visualize?)
    if not trainable:  # simply use one-hot
        emb = nn.Embedding(len(syntax_vocabulary), len(syntax_vocabulary))
        emb.from_pretrained(torch.eye(len(syntax_vocabulary)), freeze=True)
        return emb
    else:
        raise NotImplemented("No trainable embedding yet.")


class Phase(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class SpeechSyntax(Dataset):
    def __init__(self,
                 root: str, phase: Phase, syntax_vocabulary: Dict[str, int], blacklist: Set[str]):
        """
        :param root: data dir
        :param phase: train, validate, or test
        :param syntax_vocabulary: a dictionary that maps syntax tokens to their indices into the embedding
        :param blacklist: a set of sentences that will be excluded
        """
        self.phase = phase
        self.syntax_vocabulary = syntax_vocabulary
        # True if we want to return the origin audio ndarray as well;
        # otherwise, only MFCC features get returned
        self.return_original = False
        # seeded random generator used only when partitioning data
        pair_paths = [os.path.join(root, pair_path) for pair_path in os.listdir(root)]
        pair_paths = list(filter(lambda p: p not in blacklist, pair_paths))
        # shuffle deterministically
        random.Random(42).shuffle(pair_paths)
        # split data into 7: 2: 1
        if phase == Phase.TRAIN:
            self.pair_paths = pair_paths[: int(0.7 * len(pair_paths))]
        elif phase == Phase.VALIDATE:
            self.pair_paths = pair_paths[int(0.7 * len(pair_paths)): int(0.9 * len(pair_paths))]
        else:
            assert phase == Phase.TEST
            self.pair_paths = pair_paths[int(0.9 * len(pair_paths)):]

    @staticmethod
    def mfcc(inp: np.ndarray,
             n_fft: int=256, hop_length: int=80,
             n_mels: int = 26, n_mfcc: int = 13,
             ):
        """
        According to some sources(http://research.cs.tamu.edu/prism/lectures/sp/l9.pdf)
        people usually use R = 26, N = 13 for sampling rate = 8kHz
        :param inp: input in time domain, 1-D
        :param n_fft: FFT window size
        :param hop_length: how far we move the window each time
        :param n_mels: number of Mel filters
        :param n_mfcc: number of MFCC features
        :return: M(np.ndarray) [shape=(t, n_mfcc)], where t:=#(stft bins)
        # NOTE: hop_length equals 10ms -> 8000 * (10ms / 1000ms) = 80. window size is around 30ms
        """
        # form melspectrogram first
        melspectrogram = librosa.feature.melspectrogram(
            y=inp, sr=8000,  # args of librosa.feature.melspectrogram
            n_fft=n_fft, hop_length=hop_length,  # stft specification
            n_mels=n_mels,  # args of librosa.filters.mel
        )
        log_powered_melspectrogram = librosa.power_to_db(melspectrogram)
        # transposing to make first dim variable-length(spectrums of the spectrogram)
        return librosa.feature.mfcc(S=log_powered_melspectrogram, n_mfcc=n_mfcc).T

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, index) -> Tuple[Union[torch.FloatTensor, Tuple[torch.LongTensor, torch.FloatTensor]], torch.LongTensor]:
        """
        gets us
            (1) speech audio tensor(torch.FloatTensor) (dim = 1)
            (2) syntax indices(torch.LongTensor) (dim = 1)
            both have variable length
        some token sanitization also happens here
        :return: (mfcc_speech[t, n_mfcc], syntax) tuples if not return_original
        :return: ((raw_speech[N,], mfcc_speech[t, n_mfcc]), syntax) tuples if return_original
        """
        path = self.pair_paths[index]
        # handle speech
        _, raw_speech = wavfile.read(os.path.join(path, "speech.wav"))  # speech is 1-D
        mfcc_speech = self.mfcc(raw_speech.astype(dtype=np.float))
        mfcc_speech = torch.from_numpy(mfcc_speech)
        # handle syntax
        with open(os.path.join(path, "syntax.txt")) as f:
            syntax = f.read()
        # do something transformation on syntax tokens:
        # ^XP -> XP
        # XP | YP -> XP
        syntax_tokens = re.split("([\[\]])", syntax)
        syntax_tokens = list(filter(lambda t: t != "", syntax_tokens))  # remove ""
        for i in range(len(syntax_tokens)):
            st = syntax_tokens[i]
            if re.match("^\^[A-Z]+\$?$", st):
                syntax_tokens[i] = st[1:]
            elif "|" in st:
                syntax_tokens[i] = st.split("|")[0]
        syntax_idx = torch.LongTensor([self.syntax_vocabulary[st] for st in syntax_tokens])

        if self.return_original:
            return (torch.from_numpy(raw_speech), mfcc_speech), syntax_idx
        else:
            return mfcc_speech, syntax_idx


def _pack_sequences(sequences: List[torch.Tensor]) -> Tuple[nn.utils.rnn.PackedSequence, torch.LongTensor]:
    """
    packs the variable-length sequences in a batch first manner
    :param sequences:
    :return: packed sequences
    :return: INVERTED perm index to restore order later (after we are through RNN)
    """
    # We must remember the permutation index to restore the order before sorting
    lengths = torch.Tensor([seq.shape[0] for seq in sequences])
    _, perm_index = lengths.sort(dim=0, descending=True)
    # sort by length in decreasing order
    sorted_sequences_by_length = [sequences[i] for i in perm_index]
    return nn.utils.rnn.pack_sequence(sorted_sequences_by_length), invert_permutation(perm_index)


def invert_permutation(p: torch.LongTensor) -> torch.LongTensor:
    """
    CREDIT: https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    assert len(p.shape) == 1
    s = np.empty(len(p), np.long)
    s[p] = np.arange(len(p))
    return torch.LongTensor(s)


def restore_order(data: torch.Tensor, invert_perm_index: torch.LongTensor) -> torch.Tensor:
    """
    :param data: a batch-first data batch
    :param invert_perm_index: indexing to restore order before sorting by length
    :return: batch-first data with restored order
    """
    return data.index_select(dim=0, index=invert_perm_index)


def speech_syntax_collate_fn(data: List[Tuple[Union[torch.FloatTensor, Tuple[torch.LongTensor, torch.FloatTensor]], torch.LongTensor]]) \
        -> Union[Tuple[Tuple[nn.utils.rnn.PackedSequence, torch.LongTensor], Tuple[nn.utils.rnn.PackedSequence, torch.LongTensor]],
        Tuple[Tuple[nn.utils.rnn.PackedSequence, torch.LongTensor], Tuple[nn.utils.rnn.PackedSequence, torch.LongTensor], Tuple[nn.utils.rnn.PackedSequence, torch.LongTensor]]]:
    """
    The default collate_fn in DataSet can't deal with variable-length input
    In this function we'll pack it and make it palatable for RNN
    :param data: a list of (speech syntax) tuple
    :return: a batch of packed speech, a batch of packed syntax (syntax indices)
    :return: bundled with their INVERTED permutation index of sorting
    """
    if type(data[0][0]) is not tuple:
        speeches, syntaxes = zip(*data)
        return _pack_sequences(speeches), _pack_sequences(syntaxes)
    else:  # returning raw speech as well
        raw_speech = []
        mfcc_speech = []
        syntaxes = []
        for (raw, mfcc), syntax in data:
            raw_speech.append(raw)
            mfcc_speech.append(mfcc)
            syntaxes.append(syntax)
        _pack_sequences(raw_speech)
        _pack_sequences(mfcc_speech)
        _pack_sequences(syntaxes)
        return _pack_sequences(raw_speech), _pack_sequences(mfcc_speech), _pack_sequences(syntaxes)


def get_dataloader(
        root: str, phase: Phase, syntax_vocabulary: Dict[str, int], blacklist: Set[str],  # dataset-related
        batch_size: int, shuffle: bool, num_workers: int,  # dataloader-related
        return_original=False):  # miscellaneous
    """
    :return: a well-configured dataloader
    """
    dataset = SpeechSyntax(root, phase, syntax_vocabulary, blacklist)
    dataset.return_original = return_original
    return utils.data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=speech_syntax_collate_fn)
