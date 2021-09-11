import json
import logging
import os
import re

import requests
import tarfile
import zipfile
import gzip
from typing import List, Dict
from tqdm import tqdm

import gdown

from .lm_t5 import TASK_PREFIX, ADDITIONAL_SP_TOKENS
from .sentence_split import SentSplit

__all__ = ('get_dataset', 'jsonline_reader', 'jsonline_writer', 'wget')
DEFAULT_CACHE_DIR = '{}/.cache/t5qg'.format(os.path.expanduser('~'))


def jsonline_reader(filename: str):
    with open(filename, 'r') as f:
        examples = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
    return examples


def jsonline_writer(data, export):
    with open(export, 'w') as f:
        f.write('\n'.join([json.dumps(x) for x in data]))


def get_dataset(name,
                split: str = 'train',
                task_type: List or str = 'qg',
                language: List or str = 'en',
                cache_dir: str = None,
                no_prefix: bool = False,
                return_raw_triplet: bool = False):
    language = [language] if type(language) is str else language
    task_type = [task_type] if type(task_type) is str else task_type
    data = Dataset(name, cache_dir, no_prefix=no_prefix).get_data(
        split, language=language, task_type=task_type, return_raw_triplet=return_raw_triplet)
    if return_raw_triplet:
        context = [i["context"] for i in data]
        question = [i["question"] for i in data]
        answer = [i["answer"] for i in data]
        return context, question, answer
    else:
        input_texts = [i["source_text"] for i in data]
        output_texts = [i["target_text"] for i in data]
        return input_texts, output_texts


def wget(url, cache_dir: str, gdrive_filename: str = None):
    """ wget and uncompress data_iterator """
    os.makedirs(cache_dir, exist_ok=True)
    if url.startswith('https://drive.google.com'):
        assert gdrive_filename is not None, 'please provide fileaname for gdrive download'
        gdown.download(url, '{}/{}'.format(cache_dir, gdrive_filename), quiet=False)
        filename = gdrive_filename
    else:
        filename = os.path.basename(url)
        with open('{}/{}'.format(cache_dir, filename), "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    path = '{}/{}'.format(cache_dir, filename)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)
    return cache_dir


class Dataset:

    all_language_tydiqa = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 'korean', 'russian', 'swahili', 'telugu']
    all_language_alias_tydiqa = dict([(i[:2], i) for i in all_language_tydiqa])

    def __init__(self,
                 data_alias: str = 'squad',
                 cache_dir: str = None,
                 no_prefix: bool = False):
        self.data_alias = data_alias
        assert self.data_alias in ['tydiqa', 'squad']
        self.cache = '{}/data/{}'.format(DEFAULT_CACHE_DIR, self.data_alias) if cache_dir is None else cache_dir
        self.sent_splitter = SentSplit()
        self.sp_token_hl = ADDITIONAL_SP_TOKENS['hl']
        self.no_prefix = no_prefix
        logging.info('instantiate data processor')

    def get_data(self, split: str = 'train', language: List = None, task_type: List = None,
                 return_raw_triplet: bool = False):
        if self.data_alias == 'squad':
            language = [None]
        else:
            language = self.all_language_alias_tydiqa if language is None else language
        assert split in ['train', 'dev', 'test'], split
        if return_raw_triplet:
            output_prefix = '{}/{}/processed/{}.raw'.format(self.cache, self.data_alias, split)
        elif self.no_prefix:
            output_prefix = '{}/{}/processed/{}.no_prefix'.format(self.cache, self.data_alias, split)
        else:
            output_prefix = '{}/{}/processed/{}'.format(self.cache, self.data_alias, split)

        logging.info("generating examples: {}".format(split))
        full_examples = []
        for la in language:
            if self.data_alias == 'squad':
                output = '{}.jsonl'.format(output_prefix)
                path = '{}/raw/{}/{}.jsonl'.format(self.cache, self.data_alias, split)
            else:
                output = '{}.{}.jsonl'.format(output_prefix, la)
                path = '{}/raw/{}/{}.{}.jsonl'.format(self.cache, self.data_alias, self.all_language_alias_tydiqa[la], split)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # exclude YES/NO questions or unanswerable questions
            if os.path.exists(output):
                examples = jsonline_reader(output)
            else:
                if not os.path.exists(path):
                    wget('https://github.com/asahi417/t5-question-generation/releases/download/0.0.0/{}.zip'.format(self.data_alias),
                         cache_dir='{}/raw'.format(self.cache))
                data = jsonline_reader(path)
                examples = []
                for _data in tqdm(data):
                    if return_raw_triplet:
                        examples.append({i: _data[i] for i in ["context", "question", "answer"]})
                    else:
                        examples += self.process_single_data(_data)
                jsonline_writer(data=examples, export=output)
            full_examples += examples
        if task_type is None or return_raw_triplet:
            return full_examples
        else:
            assert all(i in TASK_PREFIX for i in task_type), task_type
            return [i for i in full_examples if i['task'] in task_type]

    def process_ans_ext(self, context: str, answer: str):
        sents = self.sent_splitter(context)
        ind_candidate = [n for n, s in enumerate(sents) if answer in s]
        if len(ind_candidate) == 0:
            logging.warning('answer not found \n - answer: {} \n - context: {}'.format(answer, context))
            return None
        ind = ind_candidate[-1]
        end = sum(len(i) for i in sents[:ind + 1]) + 1
        start = 0 if ind == 0 else sum(len(i) for i in sents[:ind]) + 1
        before = ' '.join(sents[:start])
        after = ' '.join(sents[end:])
        sent = sents[start:end]
        sent = "{0} {1} {2} {1} {3}".format(before, self.sp_token_hl, sent, after)
        return re.sub(r'\s+', ' ', sent)

    def process_single_data(self, data: Dict):
        """ This function returns the examples in the raw (text) form. """
        question = data["question"]
        context = data["context"]
        answer = data["answer"]
        examples = []
        if 'ans_ext' in TASK_PREFIX:
            source_text = self.process_ans_ext(context, answer)
            if source_text is not None:
                if not self.no_prefix:
                    source_text = "{}: {}".format(TASK_PREFIX['ans_ext'], source_text)
                examples.append({'source_text': re.sub(r'\s+', ' ', source_text), "target_text": answer, "task": "ans_ext"})
        if 'qa' in TASK_PREFIX:
            if self.no_prefix:
                source_text = "{} {}".format(context, question)
            else:
                source_text = "{}: {}  context: {}".format(TASK_PREFIX['qa'], question, context)
            examples.append({"source_text": re.sub(r'\s+', ' ', source_text), "target_text": answer, "task": "qa"})
        if 'qg' in TASK_PREFIX:
            position = context.find(answer)
            assert position != -1
            source_text = '{0}{1} {2} {1}{3}'.format(
                context[:position], ADDITIONAL_SP_TOKENS['hl'], answer,
                context[position + len(answer):])
            if not self.no_prefix:
                source_text = "{}: {}".format(TASK_PREFIX['qg'], source_text)
            examples.append({"source_text": re.sub(r'\s+', ' ', source_text), "target_text": question, "task": "qg"})
        return examples
