""" Sentence/question level QG evaluation. """
import argparse
import json

import qgeval_sentence_level  # for sentence-level evaluation
import nlgeval  # for question-level evaluation

REF_SENT_LEVEL = './processed'
REF_Q_LEVEL = './raw'

def nlgeval_():
    qgeval_sentence_level.text_normalization()

def get_options():
    parser = argparse.ArgumentParser(description='Sentence/question level QG evaluation.')
    parser.add_argument('-t', '--hyp-test', default=None, type=str)
    parser.add_argument('-v', '--hyp-dev', default=None, type=str)
    parser.add_argument('-e', '--export', default='metric.json', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    metrics = {}
    for split, path_h in zip(['dev', 'test'], [opt.hyp_dev, opt.hyp_test]):
        if path_h is None:
            continue
        metrics[split] = {}
        for prediction_aggregation in ['first', 'last', 'long', 'short', 'middle']:
            metrics[split]['sentence_level/{}'.format(prediction_aggregation)] = qgeval_sentence_level.compute_metrics(
                out_file=path_h,
                src_file='{}/src-{}.txt'.format(REF_SENT_LEVEL, split),
                tgt_file='{}/tgt-{}.txt'.format(REF_SENT_LEVEL, split),
                prediction_aggregation=prediction_aggregation
            )
        metrics[split]['answer_level'] = nlgeval.compute_metrics(
            hypothesis=path_h,
            references=['{}/samples.{}.ref.txt'.format(REF_Q_LEVEL, split)],
            no_skipthoughts=True,
            no_glove=True)

    print(json.dumps(metrics, indent=4, sort_keys=True))
    with open(opt.export, 'w') as f:
        json.dump(metrics, f)


