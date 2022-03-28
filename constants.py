import os

FILTERED_DATA_PATH = '/scratch/cluster/spantha/public_bug_report_data/filtered'
FULL_DATA_PATH = '/scratch/cluster/spantha/public_bug_report_data/full'

PLBART_PATH = '/scratch/cluster/spantha/PLBART-BASE/PLBART-BASE/'
SENTENCE_PIECE_MODEL_PATH = os.path.join(PLBART_PATH, 'tokenizer/sentencepiece.bpe.model')
PLBART_DICT = os.path.join(PLBART_PATH, 'dict.txt')
PLBART_CHECKPOINT = os.path.join(PLBART_PATH, 'plbart-base.pt')
SPM_PATH = '/scratch/cluster/spantha/sentencepiece/build/src/spm_encode'