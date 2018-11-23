from nlp_architect.data.intent_datasets import IntentDataset

import os
import sys
import json

class RasaNlu(IntentDataset):
    """
    RASA NLU dataset class
    Args:
            path (str): dataset path
            sentence_length (int, optional): max sentence length
            word_length (int, optional): max word length
    """

    def __init__(self, path, train_file, test_file, sentence_length=30, word_length=12):
        if path is None or not os.path.isdir(path):
            print('invalid path for RasaNlu dataset loader')
            sys.exit(0)
        self.dataset_root = path
        self.train_file = train_file
        self.test_file = test_file

        train_set_raw, test_set_raw = self._load_dataset()
        super(RasaNlu, self).__init__(sentence_length=sentence_length,
                                      word_length=word_length)
        self._load_data(train_set_raw, test_set_raw)

    def _load_dataset(self):
        train_data = self._load_intents(self.train_file)
        test_data = self._load_intents(self.test_file)
        train = [(t, l, i) for i in sorted(train_data)
                 for t, l in train_data[i]]
        test = [(t, l, i) for i in sorted(test_data) for t, l in test_data[i]]
        return train, test

    def _load_intents(self, file):
        data = {}
        fname = os.path.join(self.dataset_root, file)

        with open(fname, 'rb') as load_f:
            load_dict = json.load(load_f)
            data = load_dict['rasa_nlu_data']
            common_examples = data.get("common_examples", [])
            intent_examples = data.get("intent_examples", [])
            entity_examples = data.get("entity_examples", [])
            all_examples = common_examples + intent_examples + entity_examples

            entries = []
            sentences = []
            train_data = {}
            for ex in all_examples:
                intent = ex.get("intent")

                if intent not in train_data:
                    sentences = []

                entries = self._parse_json_jieba(ex, sentences)

                train_data[intent] = entries

        return train_data

    def _parse_json_jieba(self, data, sentences):
        import jieba
        tokens = []
        tags = ['O'] * len(data['text'])
        new_tokens = jieba.tokenize(data['text'].strip())
        tokens += [word for (word, start, end) in new_tokens]
        for s in data['entities']:

            ent = s.get('entity', None)
            start = s.get('start', None)
            end = s.get('end', None)

            tags[start:end] = self._create_tags(ent, len(s))

        sentences.append((tokens, tags))
        return sentences

    @staticmethod
    def _create_tags(tag, length):
        labels = ['B-' + tag]
        if length > 1:
            for _ in range(length - 1):
                labels.append('I-' + tag)
        return labels
