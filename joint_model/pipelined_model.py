import argparse
from datetime import datetime
import sys
import subprocess

from combined_model import *

sys.path.append('../')
from constants import *
from utils import *

s = spm.SentencePieceProcessor(model_file=SENTENCE_PIECE_MODEL_PATH)

def get_stepped_input_output(ex):
    text = []
    summary = []

    utterance_inp_list = []

    tokenized_title = [TITLE_CLS] + ex.title.tokens
    utterance_inp_list.extend(tokenized_title)
    before_code_change_turns = [ex.report] + ex.pre_utterances
    
    for u, utterance in enumerate(before_code_change_turns): 
        utterance_inp_list = utterance_inp_list + [UTTERANCE_CLS] + utterance.tokens
        text.append(utterance_inp_list)
        summary.append(ex.solution_description.tokens)

    return text, summary

def build_dataset(examples, partition):
    print('{}: Starting building {}'.format(datetime.now(), len(examples)))
    src = []
    trg = []
    for ex in examples:
        inp, out = get_stepped_input_output(ex)
        src.extend(inp)
        trg.extend(out)

    prefix = os.path.join(args.processed_data_dir, '{}'.format(partition))
    src_path = prefix + '.source'

    with open(src_path, 'w+') as f:
        for x in src:
            sent_piece = s.encode(' '.join(x))
            trunc_len = min(len(sent_piece), 1020)
            shortened_sent_piece = sent_piece[-trunc_len:]
            w_x = s.decode(shortened_sent_piece).split()
            f.write('{}\n'.format(' '.join(w_x)))

    print('Wrote {} to {}'.format(len(src), src_path))
    
    trg_path = prefix + '.target'
    with open(trg_path, 'w+') as f:
        for x in trg:
            f.write('{}\n'.format(' '.join(x)))
    print('Wrote {} to {}'.format(len(trg), trg_path))
    
    print('{}: Terminating building {}'.format(datetime.now(), len(examples)))
    return prefix

def preprocess(train_examples, valid_examples, test_examples):
    train_prefix = build_dataset(train_examples, 'train')
    valid_prefix = build_dataset(valid_examples, 'valid')
    test_prefix = build_dataset(test_examples, 'test')
    
    preprocess_arguments = [
        'sh',
        '../generation_models/plbart_scripts/plbart_preprocess.sh',
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

def get_stepped_generation_output(train_examples, valid_examples, test_examples):
    preprocess(train_examples, valid_examples, test_examples)
    generate_arguments = [
        'sh',
        '../generation_models/plbart_scripts/plbart_generate.sh',
        args.processed_data_dir,
        args.gen_model_dir,
        args.output_dir,
        '../generation_models'
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

    all_prediction_path = os.path.join(args.output_dir, 'gen_all_turns.txt')
    with open(all_prediction_path, 'w+') as f:
        for p in pred_lines:
            f.write('{}'.format(p))
    print('Wrote {} to {}'.format(len(pred_lines), all_prediction_path))

def test_pipelined_model(train_examples, valid_examples, test_examples):
    os.makedirs(args.output_dir, exist_ok=True)
    get_stepped_generation_output(train_examples, valid_examples, test_examples)
    test_classifier(test_examples)
    align_outputs(test_examples)

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
    
def test_classifier(test_examples):
    args.c_only = True
    args.model_dir = args.class_model_dir
    run_test(test_examples, args)

def train_classifier(train_examples, valid_examples):
    args.c_only = True
    args.model_dir = args.class_model_dir
    run_train(train_examples, valid_examples, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--valid_mode', action='store_true')
    parser.add_argument('--c_only', action='store_true')
    parser.add_argument('--processed_data_dir')
    parser.add_argument('--gen_model_dir')
    parser.add_argument('--class_model_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    args = parser.parse_args()

    print('{}: Loading data'.format(datetime.now()))
    train_examples, valid_examples, test_examples = get_data_splits(args.filtered)

    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))

    if args.test_mode:
        test_pipelined_model(train_examples, valid_examples, test_examples)
    elif args.valid_mode:
        test_pipelined_model(train_examples, valid_examples, valid_examples)
    else:
        train_classifier(train_examples, valid_examples)
    