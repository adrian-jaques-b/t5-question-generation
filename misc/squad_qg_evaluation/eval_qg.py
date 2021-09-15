""" SQuAD QG evaluation (sentence/answer level) """
import argparse
import json
from tools import compute_metrics


def get_options():
    parser = argparse.ArgumentParser(description='SQuAD QG evaluation (sentence/answer level)')
    parser.add_argument('-t', '--hyp-test', default=None, type=str)
    parser.add_argument('-v', '--hyp-dev', default=None, type=str)
    parser.add_argument('-e', '--export', default='metric.json', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    metrics = {'raw': {'dev': {}, 'test': {}}, 'processed': {'dev': {}, 'test': {}}}

    for split, path_h in zip(['dev', 'test'], [opt.hyp_dev, opt.hyp_test]):
        src_file = './data/src-{}-processed.txt'.format(split)

        # answer level metric
        metrics['processed'][split]['answer_level'] = compute_metrics(
            out_file=path_h, tgt_file='./data/tgt-{}-processed.txt'.format(split), normalize=True)
        metrics['raw'][split]['answer_level'] = compute_metrics(
            out_file=path_h, tgt_file='./data/tgt-{}.txt'.format(split), normalize=False)

        for prediction_aggregation in ['first', 'last', 'long', 'short', 'middle']:

            # question level metric
            metrics['processed'][split]['sentence_level/{}'.format(prediction_aggregation)] = compute_metrics(
                out_file=path_h, src_file=src_file, prediction_aggregation=prediction_aggregation,
                tgt_file='./data/tgt-{}-processed.txt'.format(split), normalize=True)
            metrics['raw'][split]['sentence_level/{}'.format(prediction_aggregation)] = compute_metrics(
                out_file=path_h, src_file=src_file, prediction_aggregation=prediction_aggregation,
                tgt_file='./data/tgt-{}.txt'.format(split), normalize=False)

    print(json.dumps(metrics, indent=4, sort_keys=True))
    with open(opt.export, 'w') as f:
        json.dump(metrics, f)


