""" SQuAD QAG evaluation """
import argparse
import json
import os
from tools import compute_metrics

from t5qg.data import get_dataset
from t5qg import T5


def get_prediction(model: str, batch_size: int = 32, max_length: int = 512, max_length_output: int = 32,
                   num_beams: int = 4):
    lm = T5(model, max_length=max_length, max_length_output=max_length_output)
    lm.eval()
    predictions = {}

    for _split in ['dev', 'test']:
        full_examples = get_dataset("squad", split=_split, return_raw_triplet=True)
        context = [i['context'] for i in full_examples]
        q_list = []
        a_list = []
        for c in context:
            qa_list = lm.generate_qa(c, num_beams=num_beams, batch_size=batch_size)
            qa_list = list(zip(qa_list))
            q_list.append(qa_list[0])
            a_list.append(qa_list[1])
        predictions[_split]['question'] = q_list
        predictions[_split]['answer'] = a_list
    return predictions


def get_options():
    parser = argparse.ArgumentParser(description='SQuAD QG evaluation (sentence/answer level)')
    parser.add_argument('-t', '--hyp-test', default=None, type=str)
    parser.add_argument('-v', '--hyp-dev', default=None, type=str)
    parser.add_argument('-e', '--export', default='metric.json', type=str)

    parser.add_argument('-m', '--model', help='pretrained language model', required=True, type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
    parser.add_argument('--num-beams', help='n  beams', default=4, type=int)
    parser.add_argument('--max-length', default=512, type=int, help='max sequence length for input sequence')
    parser.add_argument('--max-length-output', default=32, type=int, help='max sequence length for output sequence')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()

    # generate model prediction
    pred = get_prediction(
        model=opt.model, batch_size=opt.batch, max_length=opt.max_length, max_length_output=opt.max_length_output,
        num_beams=opt.num_beams)

    metrics = {'raw': {'dev': {}, 'test': {}}, 'processed': {'dev': {}, 'test': {}}}

    for split, path_h in zip(['dev', 'test'], [opt.hyp_dev, opt.hyp_test]):
        src_file = './data/src-{}-processed.txt'.format(split)

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


