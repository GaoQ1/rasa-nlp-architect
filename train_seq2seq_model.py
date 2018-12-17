from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pickle
from os import path

from tensorflow.python.keras.utils import to_categorical

from nlp_architect.contrib.tensorflow.python.keras.callbacks import ConllCallback
from nlp_architect.models.intent_extraction import Seq2SeqIntentModel
from nlp_architect.utils.io import validate, validate_existing_directory, validate_existing_filepath, validate_parent_exists
from nlp_architect.utils.metrics import get_conll_scores

from rasa_nlu_data import RasaNlu

def validate_input_args():
    global model_path
    validate((args.b, int, 1, 100000000))
    validate((args.e, int, 1, 100000000))
    validate((args.sentence_length, int, 1, 10000))
    validate((args.token_emb_size, int, 1, 10000))
    validate((args.lstm_hidden_size, int, 1, 10000))
    validate((args.encoder_depth, int, 1, 10))
    validate((args.decoder_depth, int, 1, 10))
    validate((args.encoder_dropout, float, 0, 1))
    validate((args.decoder_dropout, float, 0, 1))
    model_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_path))
    validate_parent_exists(model_path)
    model_info_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_info_path))
    validate_parent_exists(model_info_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', type=int, default=10,
                        help='Number of epochs')

    parser.add_argument('--dataset_path', type=validate_existing_directory,
                        default="rasa_data/rasa_nlu_data/", help='dataset directory')
    parser.add_argument('--training_file', type=str,
                        default="rasa_dataset_training.json", help='training data file')
    parser.add_argument('--testing_file', type=str,
                        default="rasa_dataset_testing.json", help='testing data file')
    
    
    parser.add_argument('--sentence_length', type=int, default=30,
                        help='Max sentence length')
    parser.add_argument('--token_emb_size', type=int, default=100,
                        help='Token features embedding vector size')
    parser.add_argument('--lstm_hidden_size', type=int, default=150,
                        help='Encoder LSTM hidden size')
    parser.add_argument('--encoder_depth', type=int, default=1,
                        help='Encoder LSTM depth')
    parser.add_argument('--decoder_depth', type=int, default=1,
                        help='Decoder LSTM depth')
    parser.add_argument('--encoder_dropout', type=float, default=0.5,
                        help='Encoder dropout value')
    parser.add_argument('--decoder_dropout', type=float, default=0.5,
                        help='Decoder dropout value')
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to word embedding model file')
    parser.add_argument('--model_path', type=str, default='models/seq2seq/model.h5',
                        help='Model file path')
    parser.add_argument('--model_info_path', type=str, default='models/seq2seq/model_info.dat',
                        help='Path for saving model topology')
    args = parser.parse_args()
    validate_input_args()

    dataset = RasaNlu(path=args.dataset_path,
                      train_file=args.training_file,
                      test_file=args.testing_file,
                      sentence_length=args.sentence_length)


    train_x, _, train_i, train_y = dataset.train_set
    test_x, _, test_i, test_y = dataset.test_set

    test_y = to_categorical(test_y, dataset.label_vocab_size)
    train_y = to_categorical(train_y, dataset.label_vocab_size)

    model = Seq2SeqIntentModel()
    model.build(dataset.word_vocab_size,
                dataset.label_vocab_size,
                args.token_emb_size,
                args.encoder_depth,
                args.decoder_depth,
                args.lstm_hidden_size,
                args.encoder_dropout,
                args.decoder_dropout)

    conll_cb = ConllCallback(test_x, test_y, dataset.tags_vocab.vocab, batch_size=args.b)

    # train model
    model.fit(x=train_x, y=train_y,
              batch_size=args.b, epochs=args.e,
              validation=(test_x, test_y),
              callbacks=[conll_cb])
    print('Training done.')

    print('Saving model')
    model.save(args.model_path)
    with open(args.model_info_path, 'wb') as fp:
        info = {
            'type': 'seq2seq',
            'tags_vocab': dataset.tags_vocab.vocab,
            'word_vocab': dataset.word_vocab.vocab,
            'char_vocab': dataset.char_vocab.vocab,
            'intent_vocab': dataset.intents_vocab.vocab,
        }
        pickle.dump(info, fp)

    # test performance
    predictions = model.predict(test_x, batch_size=args.b)
    eval = get_conll_scores(predictions, test_y,
                            {v: k for k, v in dataset.tags_vocab.vocab.items()})
    print(eval)
