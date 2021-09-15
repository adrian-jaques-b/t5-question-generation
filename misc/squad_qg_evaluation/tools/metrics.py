""" SQuAD QG evaluation (sentence/answer level) """
import json
from collections import defaultdict
from . import Bleu, Meteor, Rouge, text_normalization


class QGEvalCap:

    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.5f" % (m, sc))
                    output[m] = sc
            else:
                print("%s: %0.5f" % (method, score))
                output[method] = score
        return output


def compute_metrics(out_file,
                    tgt_file,
                    src_file: str = None,
                    prediction_aggregation: str = 'first',
                    normalize: bool = True):
    """

    @param out_file:
    @param src_file:
    @param tgt_file:
    @param prediction_aggregation:
    @return:
    """

    pairs = []

    with open(tgt_file, "r") as infile:
        for n, line in enumerate(infile):
            pairs.append({'tokenized_question': line[:-1].strip(), 'tokenized_sentence': n})

    if src_file is not None:
        # group by the source (sentence where the question are produced)
        with open(src_file, 'r') as infile:
            for n, line in enumerate(infile):
                pairs[n]['tokenized_sentence'] = line[:-1].strip().lower()

    # fix prediction's tokenization: lower-casing and detaching sp characters
    with open(out_file, 'r') as infile:
        for n, line in enumerate(infile):
            if normalize:
                pairs[n]['prediction'] = text_normalization(line[:-1].strip())
            else:
                pairs[n]['prediction'] = line[:-1].strip()

    # eval
    json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])

    for pair in pairs:

        # key is the sentence where the model generates the question
        key = pair['tokenized_sentence']

        # one generation per sentence
        res[key].append(pair['prediction'].encode('utf-8'))

        # multiple gold question per sentence
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    res_filtered = defaultdict(lambda: [])
    for k, v in res.items():
        if prediction_aggregation == 'first':
            # the first one
            res_filtered[k] = [v[0]]
        elif prediction_aggregation == 'last':
            # the last one
            res_filtered[k] = [v[-1]]
        elif prediction_aggregation == 'long':
            # the longest generation
            res_filtered[k] = [v[v.index(sorted(v, key=len)[-1])]]
        elif prediction_aggregation == 'short':
            # the shortest generation
            res_filtered[k] = [v[v.index(sorted(v, key=len)[0])]]
        elif prediction_aggregation == 'middle':
            # middle length generation
            res_filtered[k] = [v[v.index(sorted(v, key=len)[int(len(v)/2)])]]
        else:
            raise ValueError('unknown aggregation method: {}'.format(prediction_aggregation))

    # print(res_filtered)
    return QGEvalCap(gts, res_filtered).evaluate()

