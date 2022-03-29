import os

ROOT_DIR = '' # TODO: Absolute path to the describing-bug-report-solutions directory
SPM_PATH = 'spm_encode' # TODO: Path to the sentencepiece encoder
SENTENCE_PIECE_MODEL_PATH = 'sentencepiece.bpe.model' # TODO: Path to sentencepiece.bpe.model from here: https://github.com/wasiahmad/PLBART/tree/main/sentencepiece
PLBART_DICT = 'dict.txt' # TODO: Path to dict.txt from here: https://github.com/wasiahmad/PLBART/tree/main/sentencepiece
PLBART_CHECKPOINT = 'plbart-base.pt' # TODO: Path to plbart-base.pt from here: 

FILTERED_DATA_PATH = os.path.join(ROOT_DIR, 'public_bug_report_data/filtered')
FULL_DATA_PATH = os.path.join(ROOT_DIR, 'public_bug_report_data/full')
LABEL_DICT = os.path.join(ROOT_DIR, 'combined_systems', 'label_dict.txt')