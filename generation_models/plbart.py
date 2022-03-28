import argparse
from datetime import datetime
import os
import subprocess
import sys

import sentencepiece as spm

sys.path.append('../')
from constants import *
from utils import *

s = spm.SentencePieceProcessor(model_file=SENTENCE_PIECE_MODEL_PATH)

def format_input_output(ex):
    input_sequence = [TITLE_CLS] + ex.title.tokens
    
    before_code_change_turns = [ex.report] + ex.pre_utterances
    for utterance in before_code_change_turns:
        input_sequence.extend([UTTERANCE_CLS] + utterance.tokens)

    # Manually applying truncation to prevent fairseq-train from discarding examples
    # Truncation strategy aims to keep most recent text
    input_sequence = ' '.join(input_sequence)
    sent_piece = s.encode(input_sequence)
    trunc_len = min(len(sent_piece), 1020)
    shortened_sent_piece = sent_piece[-trunc_len:]
    input_sequence = s.decode(shortened_sent_piece)

    output_sequence = ' '.join(ex.solution_description.tokens)

    return input_sequence, output_sequence

def build_dataset(examples, partition):
    print('{}: Starting building {}'.format(datetime.now(), len(examples)))
    src = []
    trg = []
    for ex in examples:
        inp, out = format_input_output(ex)
        src.append(inp)
        trg.append(out)

    prefix = os.path.join(args.processed_data_dir, '{}'.format(partition))
    src_path = prefix + '.source'

    with open(src_path, 'w+') as f:
        for x in src:
            f.write('{}\n'.format(x))
    print('Wrote {} to {}'.format(len(src), src_path))
    
    trg_path = prefix + '.target'
    with open(trg_path, 'w+') as f:
        for x in trg:
            f.write('{}\n'.format(x))
    print('Wrote {} to {}'.format(len(trg), trg_path))
    
    print('{}: Terminating building {}'.format(datetime.now(), len(examples)))
    return prefix

def preprocess(train_examples, valid_examples, test_examples):
    print('{}: Starting Preprocessing'.format(datetime.now()))
    train_prefix = build_dataset(train_examples, 'train')
    valid_prefix = build_dataset(valid_examples, 'valid')
    test_prefix = build_dataset(test_examples, 'test')
    
    preprocess_arguments = [
        'sh',
        'plbart_scripts/plbart_preprocess.sh',
        SPM_PATH,
        SENTENCE_PIECE_MODEL_PATH,
        train_prefix,
        valid_prefix,
        test_prefix,
        args.processed_data_dir,
        PLBART_DICT
    ]

    subprocess.run(preprocess_arguments, check=True)
    print('{}: Terminating Preprocessing'.format(datetime.now()))

def train(train_examples, valid_examples):
    preprocess(train_examples, valid_examples, valid_examples)
    print('{}: Starting Training {} (Valid: {})'.format(datetime.now(), len(train_examples), len(valid_examples)))

    train_arguments = [
        'sh',
        'plbart_scripts/plbart_train.sh',
        PLBART_CHECKPOINT,
        args.processed_data_dir,
        args.model_dir,
        args.output_dir,
        '.'
    ]

    subprocess.run(train_arguments, check=True)
    print('{}: Terminating Training'.format(datetime.now()))

def test(train_examples, valid_examples, test_examples):
    preprocess(train_examples, valid_examples, test_examples)
    print('{}: Starting Evaluation {}'.format(datetime.now(), len(test_examples)))

    generate_arguments = [
        'sh',
        'plbart_scripts/plbart_generate.sh',
        args.processed_data_dir,
        args.model_dir,
        args.output_dir,
        '.'
    ]

    subprocess.run(generate_arguments, check=True)

    # Note: The output will not be in the original order
    with open(os.path.join(args.output_dir, 'hypotheses.txt')) as f:
        pred_lines = f.readlines()
    
    with open(os.path.join(args.output_dir, 'target.txt')) as f:
        trg_lines = f.readlines()
    
    with open(os.path.join(args.output_dir, 'order.txt')) as f:
        order_lines = f.readlines()
    
    order = []
    for l in order_lines:
        order.append(int(l.strip().split('-')[-1]))
    
    predictions = []
    references = []

    reordered_pred_lines = ['' for _ in range(len(order))]
    reordered_trg_lines = ['' for _ in range(len(order))]
    for o, o_idx in enumerate(order):
        reordered_pred_lines[o_idx] = pred_lines[o]
        reordered_trg_lines[o_idx] = trg_lines[o]
    
    pred_lines = reordered_pred_lines
    trg_lines = reordered_trg_lines

    for e, ex in enumerate(test_examples):
        pred_line = pred_lines[e].strip()
        predictions.append(pred_line.split())
        references.append([ex.solution_description.tokens])

        print(ex.issue_url)
        print('Title: {}'.format(' '.join(ex.title.tokens)))
        print('Gold: {}'.format(' '.join(ex.solution_description.tokens)))
        print('Prediction: {}'.format(pred_line))
        print('--------------------------------')

    scores = compute_scores(references, predictions)
    for metric, vals in scores.items():
        print('{}: {}'.format(metric, 100*sum(vals)/float(len(vals))))
    print('--------------------------------')
    print('Total: {}'.format(len(order)))

    print('{}: Terminating Evaluation'.format(datetime.now()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--valid_mode', action='store_true')
    parser.add_argument('--processed_data_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    print('{}: Loading data'.format(datetime.now()))
    train_examples, valid_examples, test_examples = get_data_splits(args.filtered)
    
    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))

    if args.test_mode:
        test(train_examples, valid_examples, test_examples)
    elif args.valid_mode:
        test(train_examples, valid_examples, valid_examples)
    else:
        train(train_examples, valid_examples)