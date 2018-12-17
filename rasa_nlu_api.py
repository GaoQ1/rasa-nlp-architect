import numpy as np
import pickle
import jieba

from os import makedirs, path, sys

from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.intent_extraction import MultiTaskIntentModel, Seq2SeqIntentModel
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import download_unlicensed_file

class RasaNluApi(AbstractApi):
    def __init__(self, model_path, model_info_path):
        self.model = None
        self.model_path = model_path
        self.model_info_path = model_info_path
        with open(self.model_info_path, 'rb') as fp:
            model_info = pickle.load(fp)
        self.model_type = model_info['type']
        self.word_vocab = model_info['word_vocab']
        self.tags_vocab = {v: k for k, v in model_info['tags_vocab'].items()}
        if self.model_type == 'mtl':
            self.char_vocab = model_info['char_vocab']
            self.intent_vocab = {v: k for k,
                                 v in model_info['intent_vocab'].items()}

    def process_text(self, text):
        return [t for (t, _, _) in jieba.tokenize(text)]

    def bio_to_spans(self, text, tags):
        pointer = 0
        starts = []
        for i, t, in enumerate(tags):
            if t.startswith('B-'):
                starts.append((i, pointer))
            pointer += len(text[i])

        spans = []
        for s_i, s_char in starts:
            label_str = tags[s_i][2:]

            e = 0
            e_char = len(text[s_i + e])

            while len(tags) > s_i + e + 1 and tags[s_i + e + 1].startswith('I-'):
                e += 1
                e_char += len(text[s_i + e])
            spans.append((s_char, s_char + e_char, label_str))

        return spans

    def display_results(self, text_str, predictions, intent_type): # todo calc confidence
        text = ''.join([t for t in text_str])
        ret = {
            'intent': {},
            'entities': [],
            'text': text
        }

        spans = []
        for s, e, tag in self.bio_to_spans(text_str, predictions):
            spans.append({
                'start': s,
                'end': e,
                'entity': tag,
                'value': text[s:e],
                'confidence': 1
            })

        ret['entities'] = spans
        ret['intent'] = {
            "name": intent_type,
            "confidence": 1
        }

        return ret

    def vectorize(self, doc, vocab, char_vocab=None):
        words = np.asarray([vocab[w.lower()] if w.lower() in vocab else 1 for w in doc])\
            .reshape(1, -1)
        if char_vocab is not None:
            sentence_chars = []
            for w in doc:
                word_chars = []
                for c in w:
                    if c in char_vocab:
                        _cid = char_vocab[c]
                    else:
                        _cid = 1
                    word_chars.append(_cid)
                sentence_chars.append(word_chars)
            sentence_chars = np.expand_dims(pad_sentences(sentence_chars, self.model.word_length),
                                            axis=0)
            return [words, sentence_chars]
        else:
            return words

    def inference(self, doc):
        text_arr = self.process_text(doc)
        intent_type = None
        if self.model_type == 'mtl':
            doc_vec = self.vectorize(
                text_arr, self.word_vocab, self.char_vocab)
            intent, tags = self.model.predict(doc_vec, batch_size=1)
            intent = int(intent.argmax(1).flatten())
            intent_type = self.intent_vocab.get(intent, None)
        else:
            doc_vec = self.vectorize(text_arr, self.word_vocab, None)
            tags = self.model.predict(doc_vec, batch_size=1)
        tags = tags.argmax(2).flatten()
        tag_str = [self.tags_vocab.get(n, None) for n in tags]
        return self.display_results(text_arr, tag_str, intent_type)

    def load_model(self):
        if self.model_type == 'seq2seq':
            model = Seq2SeqIntentModel()
        else:
            model = MultiTaskIntentModel()
        model.load(self.model_path)
        self.model = model
