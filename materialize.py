"""
This script takes the output of linearize.py
and materialize
    (1) linearized syntax trees
    (2) speech wav files
"""

import re
import json
from typing import *
import os
from linearize import Sentence
import numpy as np
from scipy.io import wavfile as wf

import glob

from settings import *


class Pairer:
    def __init__(self, tree_db: Dict[str, Dict[str, Dict]], data_dir: str):
        self.tree_db = tree_db
        self.data_dir = data_dir
        self._cache: Dict[str, (int, np.ndarray)] = {}  # caches the sphfile read

    def write_paired_data(self, observation_id: str, sentence_id) -> str:
        """
        write the wav and linearized syntax tree into out_dir
        :param observation_id: e.g. sw2005
        :param sentence_id: e.g s3_500
        :return: the subdir under data_dir where speech wav and linearized syntax are stored
        """
        out_dir = os.path.join(self.data_dir, "{}#{}".format(observation_id, sentence_id))  # e.g. sw2005#s2_7
        if not os.path.exists(out_dir):  # create the subdir if it's not there already
            os.system("mkdir {}".format(out_dir))
        sentence_dict = self.tree_db[observation_id][sentence_id]
        sentence = Sentence(sentence_dict["linearized_tree"],
                            float(sentence_dict["start"]), float(sentence_dict["end"]), sentence_dict["ab"])
        rate_and_data = self._cache.get(observation_id, None)  # use cache to minimize sph reading
        if rate_and_data is None:
            rate_and_data = self._load_wavfile(observation_id)
            self._cache[observation_id] = rate_and_data
        rate, data = rate_and_data
        # write wav
        # TODO: make sure that this is the right way to slice
        start_frame, end_frame = int(sentence.start * rate), int(sentence.end * rate)
        assert start_frame < end_frame
        # TODO: make sure it's okay to split A, B channels this way
        wf.write(os.path.join(out_dir, "speech.wav"), rate, data[start_frame: end_frame, 0 if sentence.ab == "A" else 1])
        # write linearized syntax tree
        with open(os.path.join(out_dir, "syntax.txt"), mode="w") as f:
            f.write(sentence.linearized_tree)
        print("Written wav and txt under {}".format(out_dir))
        return out_dir

    def check(self, observation_id: str, sentence_id) -> Tuple[str, str]:
        """
        :param observation_id:
        :param sentence_id:
        :return: speech wav path, syntax text
        """
        out_dir = os.path.join(self.data_dir, "{}#{}".format(observation_id, sentence_id))  # e.g. sw2005#s2_7
        with open(os.path.join(out_dir, "syntax.txt")) as f:
            return os.path.join(out_dir, "speech.wav"), f.read()

    @staticmethod
    def _load_wavfile(observation_id: str) -> Tuple[int, np.ndarray]:
        ob_id = re.search("sw([0-9]+)", observation_id).group(1)  # sw2005 -> 2005
        swbd_path = os.path.join(SWBD_ROOT, "**", "sw*{}.wav".format(ob_id))
        filenames = glob.glob(swbd_path, recursive=True)
        if len(filenames) != 1:
            print("WARNING: Multiple or no WAV files associated with {}".format(observation_id))
            exit(1)
        filename = filenames[0]
        print("Found WAV File: {}".format(filename))
        rate, data = wf.read(filename)
        return rate, data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='materialize data')
    parser.add_argument("data_dir", nargs='?', default="data")
    args = parser.parse_args()

    data_dir = args.data_dir
    # check and see if a directory is already there
    ext = os.path.exists(data_dir)

    if ext:
        print("Action Aborted because there's already a directory named {}".format(data_dir))
        exit(0)
    else:
        print("creating new directory called {}".format(data_dir))
        os.system("mkdir {}".format(data_dir))

    # I decided to put a pair of parallel data into the same folder
    # named as {observation_id}#{sentence_id}

    import sys
    tree_db = json.load(sys.stdin)
    pairer = Pairer(tree_db, data_dir)
    for observation_id, sentence_table in tree_db.items():
        for sentence_id in sentence_table:
            pairer.write_paired_data(observation_id, sentence_id)
