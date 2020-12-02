import os
import csv
import logging
import random

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset 
import json
import re

def preprocess_text(text):
    text = text.replace("/", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("|", "")
    text = text.replace("~", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace("…", "")
    text = ' '.join(text.split())  # 공백이 여러개인 경우를 없애기 위함
    # text = re.sub(r'[a-zA-Z0-9]+', r'', text)  # 알파벳 및 숫자 제거
    return text

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, label=None, fn=None):
        """Constructs a InputExample/

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.fn = fn

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of 'InputExample's for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of 'InputExample's for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class AugProcessor(DataProcessor):
    """Processor for dataset to be augmented."""

    def get_train_examples(self, data_dir):
        """See base calss."""
        return self._create_examples2("train")
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_labels(self, name):
        """add your dataset here"""
        if name in ['stsa.binary', 'mpqa', 'rt-polarity', 'subj']:
            return ["0", "1"]
        elif name in ['stsa.fine']:
            return ["0", "1", "2", "3", "4"]
        elif name in ['TREC']:
            return ["0", "1", "2", "3", "4", "5"]
        elif name in ['2020AIGrand']:
            return [0, 1, 2, 3, 4]
    
    def _create_examples(self, lines, set_type):
        """Create examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples
    
    def _create_examples2(self, set_type):
        self.class2ans = {
            '000001': 0,
            '020121': 1,
            '02051': 2,
            '020811': 3,
            '020819': 4
        }
        self.pat = re.compile('[A-Z]\s?:\s?(.*)')
        self.pat1 = re.compile('(.*)')

        with open('datasets/add_v7_train.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open('datasets/add_v7_dev.json', 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        self.data = data + data2
        examples = []

        for idx, conv in enumerate(self.data):
            text = []
            for line in conv['text']:
                if self.pat.search(line):
                    line_ = self.pat.search(line).group(1)
                else:
                    line_ = self.pat1.search(line).group(1)
                
                if line_:
                    text.append(line_)

            text = map(preprocess_text, text)
            text = ' '.join(text)

            label = self.class2ans[conv['label']]

            guid = "%s-%s" % (set_type, idx)
            examples.append(
                InputExample(guid=guid, text_a=text, label=label, fn=conv['file_name']))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of 'InputBatch's."""
    
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    
    features = []
    for (ex_index, example) in enumerate(examples):
        # The convention in BERT is:
        # tokens:   [CLS] is this jack ##son ##ville ? [SEP]
        # type_ids: 0     0  0    0    0     0       0 0    
        tokens_a = tokenizer._tokenize(example.text_a)
        tokens_label = label_map[example.label]
        tokens, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels = \
            extract_features(tokens_a, tokens_label, max_seq_length, tokenizer)
        
        """convert label to label_id"""
        label_id = label_map[example.label]

        """consturct features"""
        features.append(
            InputFeature(
                init_ids=init_ids,        
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                masked_lm_labels=masked_lm_labels,
                fn=example.fn,
                label=example.label))

        """print examples"""
        if ex_index < 5:
            logger.info("[cbert] *** Example ***")
            logger.info("[cbert] guid: %s" % (example.guid))
            logger.info("[cbert] file_name: %s" % (example.fn))
            logger.info("[cbert] lebel: %s" % (example.label))
            logger.info("[cbert] tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("[cbert] init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("[cbert] input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("[cbert] input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("[cbert] segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("[cbert] masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
    return features

class BaseDataset(Dataset):
    def __init__(self, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels, fns, labels):
        self.tokenizer = tokenizer
        self.args = args
        self.sample_counter = 0
        self.class2ans = {
            '000001': 0,
            '020121': 1,
            '02051': 2,
            '020811': 3,
            '020819': 4
        }
        self.pat = re.compile('[A-Z]\s?:\s?(.*)')
        self.pat1 = re.compile('(.*)')

        with open('datasets/add_v7_train.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open('datasets/add_v7_dev.json', 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        self.data = data + data2

        for idx, conv in enumerate(self.data):
            text = []
            for line in conv['text']:
                if self.pat.search(line):
                    line_ = self.pat.search(line).group(1)
                else:
                    line_ = self.pat1.search(line).group(1)
                
                if line_:
                    text.append(line_)

            text = map(preprocess_text, text)
            text = ' '.join(text)

            self.data[idx]['text'] = preprocess_text
            self.data[idx] = self.class2ans[conv['label']]

    def __getitem__(self, idx):
        guid = self.sample_counter
        self.sample_counter += 1

        t1 = self.data[idx]['text']
        label = self.data[idx]['label']
        fn = self.data[idx]['file_name']

        tokens_a = self.tokenizer.tokenize(t1)

        example = InputExample(guid, tokens_a, label, fn)

        features = extract_features(example, label, self.args.max_seq_length, self.tokenizer)

        input = {
            'input_ids':features.input_ids,
            'input_mask':features.input_mask,
            'segment_ids':features.segment_ids,
            'mlm_label_ids': features.mlm_label_ids
        }

        for k, v in input.items():
            input[k] = torch.tensor(v)

        return input['input_ids'], input['input_mask'], input['segment_ids'], input['mlm_label_ids'], fn, label
    
    def __len__(self):
        return len(self.data)

def extract_features(example, sent_label, max_seq_length, tokenizer):
    """extract features from tokens"""
    tokens_a = example.tokans_a
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]
    
    tokens_a, tokens_a_label = create_masked_lm_predictions(tokens_a, tokenizer)
    mlm_label_ids = ([-100] + tokens_a_label + [-100])

    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(sent_label) # add label info
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(sent_label)
    tokens.append('[SEP]')
    segment_ids.append(sent_label)

    ## construct init_ids for each example
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        mlm_label_ids.append(-100)
   
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(mlm_label_ids) == max_seq_length

    if example.guid < 5:
        logger.info("[cbert] *** Example ***")
        logger.info("[cbert] guid: %s" % (example.guid))
        logger.info("[cbert] file_name: %s" % (example.fn))
        logger.info("[cbert] lebel: %s" % (example.label))
        logger.info("[cbert] tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        logger.info("[cbert] input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("[cbert] input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("[cbert] segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("[cbert] masked_lm_labels: %s" % " ".join([str(x) for x in mlm_label_ids]))


    features = InputFeature(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             mlm_label_ids=mlm_label_ids)

    return features


def create_masked_lm_predictions(tokens, tokenizer):
    """Creates the predictions for the masked LM objective."""

    output_label = []

    for idx, token in enumerate(tokens):
        prob = random.random()

        if prob < 0.15:
            prob /= 0.15

            if prob < 0.8:
                tokens[idx] = '[MASK]'
            
            elif prob < 0.9:
                tokens[idx] = random.choice(list(tokenizer.vocab.items()))[0]

            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                output_label.append(tokenizer.vocab['[UNK]'])
                logger.warning(f'Cannot find token "{token}" in vocab. Using [UNK] instead')
        else:
            output_label.append(-100)

    return tokens, output_label