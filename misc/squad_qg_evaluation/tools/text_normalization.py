import re


def text_normalization(text):
    _tok_dict = {"(": "-lrb-", ")": "-rrb-", "[": "-lsb-", "]": "-rsb-", "{": "-lcb-", "}": "-rcb-"}

    # lower casing
    text = text.lower()

    # convert into the representation of the gold question
    for k, v in _tok_dict.items():
        text = text.replace(k, v)

    # add half space before eg) What? --> What ?
    for s in ["'s", "?", ",", "'", "$"]:
        text = re.sub(r'\s*\{}'.format(s), ' {}'.format(s), text)

    # more than 2 continuation, add half space eg) A... --> A ...
    for s in [".", "-"]:
        text = re.sub(r'\s*([{}].{{2}})'.format(s), r' \1', text)

    return text

