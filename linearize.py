"""
This script deals with trees
"""

import os
import re
from xml.etree import ElementTree as ET
from typing import *
from collections import defaultdict
import sys

from settings import *


class Sentence:
    def __init__(self, linearized_tree: str, start: float, end: float, ab: str):
        """
        :param ab: whether it's speaker A or B
        """
        self.linearized_tree = linearized_tree
        self.start = start
        self.end = end
        self.ab = ab


class Word:
    def __init__(self, orth, pos):
        self.orth = orth
        self.pos = pos


def linearize(tree: ET.Element, lexicalized: bool, term_db: Dict[str, Dict[str, Word]]) -> str:
    """
    Grammar as a Foreign Language
    https://arxiv.org/pdf/1412.7449.pdf
    https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/grammar-as-a-foreign-language.md
    :param tree: a tree rooted by "S"
    :param lexicalized: true if we want to keep the text
    :param term_db: a lookup table for terminals
    :return: linearized parse tree
    """
    if tree.tag == "{http://nite.sourceforge.net/}child":
        href = tree.attrib["href"]
        s = re.search("(sw[0-9]+).[AB].terminals.xml#id\((.*?)\)", href)
        observation_id, terminal_id = s.group(1), s.group(2)
        if terminal_id in term_db[observation_id]:
            word = term_db[observation_id][terminal_id]
        else:
            print("WARNING: word discrepancy on {}, {}".format(observation_id, terminal_id), file=sys.stderr)
            word = Word("<UNK>", "UNK")
        if not lexicalized:
            return "[{}]".format(word.pos)
        else:
            return "[{}[{}]]".format(word.pos, word.orth)
    else:
        s = "[" + tree.attrib["cat"]
        for child in tree:
            s += linearize(child, lexicalized, term_db)
        s += "]"
        return s


def make_terminal_db():
    """
    builds up a dictionary:
        {
            observation_id: {
                terminal_id: word
            }
        }
    :return: tree lookup table
    """
    d = defaultdict(dict)
    path = os.path.join(NXT_ROOT, "xml", "terminals")
    ob_filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for ob_filename in ob_filenames:
        print("term_db: ", ob_filename, file=sys.stderr)
        s = re.search("(sw[0-9]+).[AB].terminals.xml", ob_filename)
        ob_name = s.group(1)
        niteid = "{http://nite.sourceforge.net/}id"

        tree = ET.parse(ob_filename).getroot()
        for term in tree:
            terminal_id = term.attrib[niteid]
            if term.tag == "word":
                d[ob_name][terminal_id] = Word(term.attrib["orth"], term.attrib["pos"])
            elif term.tag == "sil":
                d[ob_name][terminal_id] = Word("<SIL>", "SIL")
            elif term.tag == "punc":
                d[ob_name][terminal_id] = Word("<PUNC>", "PUNC")
            elif term.tag == "trace":
                d[ob_name][terminal_id] = Word("<TRACE>", "TRACE")
            else:
                raise Exception("Unknown Tag: " + term.tag)
    return dict(d)


def make_tree_bank(lexicalized: bool) -> Dict[str, Dict[str, Dict]]:
    """
    builds up a dictionary:
        {
            observation_id: {
                sentence_id: Sentence
            }
        }
    :return: tree lookup table
    """
    term_db = make_terminal_db()
    d = defaultdict(dict)
    path = os.path.join(NXT_ROOT, "xml", "syntax")
    ob_filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for ob_filename in ob_filenames:
        print("stx_db: ", ob_filename, file=sys.stderr)
        s = re.search("(sw[0-9]+).([AB]).syntax.xml", ob_filename)
        ob_name, ab = s.group(1), s.group(2)
        niteid = "{http://nite.sourceforge.net/}id"

        tree = ET.parse(ob_filename).getroot()
        assert tree
        for parse_tree in tree:
            assert parse_tree.tag == "parse"
            for root_nt in parse_tree:
                assert root_nt.tag == "nt"
                if root_nt.attrib.get("cat") == "S":  # found a sentence
                    sentence_id = root_nt.attrib.get(niteid)
                    start = root_nt.attrib.get("{http://nite.sourceforge.net/}start", None)
                    end = root_nt.attrib.get("{http://nite.sourceforge.net/}end", None)
                    if start is None or end is None:
                        print("WARNING: untimed sentence in {}, missing field".format(ob_name), file=sys.stderr)
                        continue
                    try:
                        float(start)
                        float(end)
                    except ValueError:
                        print("WARNING: untimed sentence in {}, bad float".format(ob_name), file=sys.stderr)
                        continue
                    d[ob_name][sentence_id] = Sentence(linearize(root_nt, lexicalized, term_db),
                                                       float(start), float(end), ab).__dict__
    return dict(d)


if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser(description='build syntax lookup table.')
    parser.add_argument('--lex', dest='lex', action='store_true',
                        help='build lexicalized tree')
    parser.set_defaults(lex=False)

    args = parser.parse_args()

    lex = args.lex
    d = make_tree_bank(lex)
    json.dump(d, sys.stdout)
