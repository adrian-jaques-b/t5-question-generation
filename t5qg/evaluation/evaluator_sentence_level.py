import json
from collections import defaultdict

from .text_normalization import text_normalization

from .evaluator_sentence_level_tools.bleu.bleu import Bleu
from .evaluator_sentence_level_tools.meteor.meteor import Meteor
from .evaluator_sentence_level_tools.rouge import Rouge

__all__ = 'compute_metrics'


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


def compute_metrics(hypothesis_file, source_file, reference_file, prediction_aggregation: str = 'first'):
    """ """

    pairs = []
    with open(source_file, 'r') as infile:
        for line in infile:
            pairs.append({'tokenized_sentence': line[:-1].strip().lower()})

    with open(reference_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1].strip()
            cnt += 1

    # fix prediction's tokenization: lower-casing and detaching sp characters
    output = []
    with open(hypothesis_file, 'r') as infile:
        for line in infile:
            line = text_normalization(line[:-1].strip())
            output.append(line)

    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    # eval
    json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])

    for pair in pairs[:]:

        # key is the sentence where the model generates the question
        key = pair['tokenized_sentence']

        # one generation per sentence
        # res[key] = [pair['prediction'].encode('utf-8')]
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

    return QGEvalCap(gts, res_filtered).evaluate()
