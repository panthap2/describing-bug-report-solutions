import argparse
from datetime import datetime
import sys
import subprocess

from combined_model import *

sys.path.append('../')
from constants import *
from utils import *

def test_jointly_trained_model(test_examples):
    os.makedirs(args.output_dir, exist_ok=True)
    args.c_only = False
    run_test(test_examples, args)
    align_outputs(test_examples)

def train_jointly_trained_model(train_examples, valid_examples):
    args.c_only = False
    run_train(train_examples, valid_examples, args)

def align_outputs(test_examples):
    class_file = os.path.join(args.output_dir, 'class.txt')
    gen_file = os.path.join(args.output_dir, 'gen_all_turns.txt')

    with open(class_file) as f:
        class_lines = f.readlines()

    assert len(class_lines) == len(test_examples)

    with open(gen_file) as f:
        gen_lines = f.readlines()
    
    gen_offset = 0
    total_refrained = 0
    distances = []
    predictions = []
    references = []
    for e, ex in enumerate(test_examples):
        before_turns = [ex.report] + ex.pre_utterances
        gold_idx = len(before_turns)-1
        pred_idx = int(class_lines[e].strip())

        if pred_idx == -1:
            total_refrained += 1
        else:
            distances.append(gold_idx-pred_idx)
            pred_description = gen_lines[gen_offset+pred_idx].strip()
            predictions.append(pred_description.split())
            references.append([ex.solution_description.tokens])
        
        print(ex.issue_url)
        print('Title: {}'.format(' '.join(ex.title.tokens)))
        print('Gold ({}): {}'.format(gold_idx, ' '.join(ex.solution_description.tokens)))
        if pred_idx != -1:
            print('Prediction ({}): {}'.format(pred_idx, pred_description))
        print('--------------------------------')
        
        gen_offset += len(before_turns)
    
    print('\nGeneration Performance:')
    scores = compute_scores(references, predictions)
    for metric, vals in scores.items():
        print('{}: {}'.format(metric, 100*sum(vals)/float(len(vals))))
    print('--------------------------------')
    print('\nClassification Performance:')
    print('Refrained: {}%'.format(100.0*(total_refrained/float(len(test_examples)))))
    print('Avg distance: {}'.format(float(sum(distances))/len(distances)))
    print('--------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--valid_mode', action='store_true')
    parser.add_argument('--c_only', action='store_true')
    parser.add_argument('--processed_data_dir')
    parser.add_argument('--gen_model_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    args = parser.parse_args()

    print('{}: Loading data'.format(datetime.now()))
    train_examples, valid_examples, test_examples = get_data_splits(args.filtered)

    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))

    if args.test_mode:
        test_jointly_trained_model(test_examples)
    elif args.valid_mode:
        test_jointly_trained_model(valid_examples)
    else:
        train_jointly_trained_model(train_examples, valid_examples)
    