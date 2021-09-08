# QG Evaluation Tools on SQuAD
We provide a script to compute sentence-level metric and question-level metric.
The sentence-level metric should be compatible with [the existing baselines](https://paperswithcode.com/sota/question-generation-on-squad11).
```shell
python eval.py
```
```shell
usage: eval.py [-h] [-t HYP_TEST] [-v HYP_DEV] [-e EXPORT]

Sentence/question level QG evaluation.

optional arguments:
  -h, --help            show this help message and exit
  -t HYP_TEST, --hyp-test HYP_TEST
  -v HYP_DEV, --hyp-dev HYP_DEV
  -e EXPORT, --export EXPORT
```
It exports a result json file that has following structure. 
```json
{
  "dev": {
    "sentence_level/first": {
      "Bleu_1": 0.5718684448388638,
      "Bleu_2": 0.4201765812690984,
      "Bleu_3": 0.33033083950996234,
      "Bleu_4": 0.26665105139963186,
      "METEOR": 0.27020808327745766,
      "ROUGE_L": 0.5413544277532728
    },
   "sentence_level/last": {...},
   "sentence_level/long": {...},
   "sentence_level/short": {...},
   "sentence_level/middle": {...},
   "question_level": {...}
  },
 "test": {...}
}
```
Sentence-level metric has some variation depending on the hypothesis selection type.

## What is Sentence/Question Level Metric?
Question generation (QG) model evaluation on SQuAD usually follows the 
[original Neural Question Generation (NQG) paper](https://arxiv.org/pdf/1705.00106.pdf)
where the model prediction on a single sentence is evaluated on a list of gold questions for the same sentence.
For example, with the sentence below,
```
It was founded in 1986 through the donations of Joan B. Kroc , the widow of McDonald's owner Ray Kroc. 
```
SQuAD provides three questions. 
```
In what year was the Joan B. Kroc institute for international peace studies founded?
To whom was John B. Kroc married?
What company did Ray Kroc own?
```
However, since the original NQG model was only conditioned by the sentence, meaning the prediction is per-sentence level, 
the evaluation becomes sentence level rather than question level.
Concretely, all the gold questions associated with a unique sentence are regarded as the references,
and sequence evaluation metrics such as BLEU/ROUGE/METEOR are computed with them.
Note that current SotA models such as [ERNIE-gen](https://arxiv.org/pdf/2001.11314.pdf) also follows this pipeline to evaluate the model 
so that it can be compared with the baselines including NQG. 

### Selectivity of Hypothesis
Unlike NQG, neural question generation models can also be conditioned by the answer in the sentence. For example, 
[BERT-QG](https://aclanthology.org/D19-5821.pdf) proposes to condition the input sentence with a highlight token around the answer. 
```
It was founded in [HL] 1986 [HL] through the donations of Joan B. Kroc , the widow of McDonald's owner Ray Kroc. 
```
With such a modification, model can produce different predictions within a single sentence depending on the answer. In such a case, the selection 
of the model hypothesis would change the final metric in the sentence-level evaluation.
For example, in the ERNIE's official implementation, it seems that they just regard the last prediction as the model hypothesis 
(see [here](https://github.com/PaddlePaddle/ERNIE/blob/repro/ernie-gen/eval/tasks/squad_qg/qg/eval_on_unilm_tokenized_ref.py#L200)),
but any selection criteria can be used such as choosing the longest or shortest prediction.

### Normalization Effect in Evaluation
The references of SQuAD used in NQG paper are lower-cased and normalized before the evaluation, that look like below.
```
in what year was the joan b. kroc institute for international peace studies founded ?
to whom was john b. kroc married ?
what company did ray kroc own ?
```
In our evaluation script, we normalize the model prediction following NQG pipeline when we compute sentence-level metric but the question-level metric relies on raw references.

### Question Level Metric
If the model is conditioned by the answer and can generate diverse question within a sentence, we can conduct question-level metric. Essentially, 
we can compute the sequence metric per question, and this should be more challenging yet realistic metric than the sentence-level metric.
In our script, we don't apply the NQG's normalization and compute the metric on the raw reference.

## Reference
- Normalized dataset: [test](./processed/tgt-test.txt), [validation](./processed/tgt-dev.txt)
- Raw dataset: [test](./raw/samples.test.ref.txt), [validation](./raw/samples.dev.ref.txt)

