from io import StringIO
import json
import os
import re
import subprocess
import sys
from typing import Optional, Tuple
import torch

import files2rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.meteor.meteor import Meteor

from constants import *

MARKERS = ['[CODE-BLOCK]', '[/CODE-BLOCK]', '[USER-MENTION-BLOCK]',
    '[/USER-MENTION-BLOCK]', '[IMAGE-BLOCK]', '[URL-BLOCK]', '[/URL-BLOCK]']
SPECIAL_BLOCK_REGEX = ''
for m, marker in enumerate(MARKERS):
    SPECIAL_BLOCK_REGEX += '\{}\{}'.format(marker[:-1], marker[-1])

    if m < len(MARKERS) - 1:
        SPECIAL_BLOCK_REGEX += '|'

SPECIAL_BLOCK_REGEX = '(\S*)({})(\S*)'.format(SPECIAL_BLOCK_REGEX)

UTTERANCE_CLS = '<U_START>'
TITLE_CLS = '<T_START>'
SOS = '<SOS>'
EOS = '<EOS>'
LENGTH_CUTOFF_PCT = 95
VOCAB_CUTOFF_PCT = 5
MAX_VOCAB_SIZE = 50000
PATIENCE = 5
BEAM_SIZE = 2 #5
NUM_LAYERS = 2

class IssueData:
    def __init__(self, id, issue_url, title, code_change_url, solution_description, 
                 report, pre_utterances, post_utterances, start_date_str, end_date_str,
                 project, labels):
        self.id = id
        self.issue_url = issue_url
        self.title = title
        self.code_change_url = code_change_url
        self.solution_description = solution_description
        self.report = report
        self.pre_utterances = pre_utterances
        self.post_utterances = post_utterances
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.project = project
        self.labels = labels
    
    def to_json(self):
        return {
            'id': self.id,
            'issue_url': self.issue_url,
            'title': self.title.to_json(),
            'code_change_url': self.code_change_url,
            'solution_description': self.solution_description.to_json(),
            'report': self.report.to_json(),
            'pre_utterances': [u.to_json() for u in self.pre_utterances],
            'post_utterances': [u.to_json() for u in self.post_utterances],
            'start_date_str': self.start_date_str,
            'end_date_str': self.end_date_str,
            'project': self.project,
            'labels': self.labels,
        }
    
    @classmethod
    def from_json(cls, obj):
        return IssueData(
            obj['id'],
            obj['issue_url'],
            TextInstance.from_json(obj['title']),
            obj['code_change_url'],
            TextInstance.from_json(obj['solution_description']),
            TextInstance.from_json(obj['report']),
            [TextInstance.from_json(u) for u in obj['pre_utterances']],
            [TextInstance.from_json(u) for u in obj['post_utterances']],
            obj['start_date_str'], obj['end_date_str'], obj['project'],
            obj['labels'],
        )

class TextInstance:
    def __init__(self, author_index, processed_text, tokens):
        self.author_index = author_index
        self.processed_text = processed_text
        self.tokens = tokens
    
    def to_json(self):
        return {
            'author_index': self.author_index,
            'processed_text': self.processed_text,
            'tokens': self.tokens
        }
    
    @classmethod
    def from_json(cls, obj):
        return TextInstance(
            obj['author_index'],
            obj['processed_text'],
            obj['tokens']
        )

def get_data_splits(filtered=False):
    def load_file(fpath):
        with open(fpath) as f:
            data = json.load(f)
        
        return [IssueData.from_json(d) for d in data]

    if filtered:
        data_path = FILTERED_DATA_PATH
    else:
        data_path = FULL_DATA_PATH
    
    train_examples = load_file(os.path.join(data_path,'train.json'))
    valid_examples = load_file(os.path.join(data_path,'valid.json'))
    test_examples = load_file(os.path.join(data_path,'test.json'))

    return train_examples, valid_examples, test_examples

def compute_scores(reference_list, sentences):
    preds = dict()
    refs = dict()

    for i in range(len(sentences)):
        preds[i] = [' '.join([s for s in sentences[i]])]
        refs[i] = [' '.join(l) for l in reference_list[i]]

    final_scores = dict()

    scorers = [
        (Meteor(),"METEOR"),
    ]

    final_scores = dict()
    final_scores['NLTK BLEU-4'] = compute_bleu(reference_list, sentences)

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = scs
        else:
            final_scores[method] = scores
    
    final_scores.update(compute_fine_rouge(reference_list, sentences))

    return final_scores

def compute_bleu(references, hypotheses):
    bleu_4_sentence_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu_4_sentence_scores.append(sentence_bleu(ref, hyp,
            smoothing_function=SmoothingFunction().method2))
    return bleu_4_sentence_scores

def compute_fine_rouge(reference_list, sentences):
    candidates = []
    references = []

    for i in range(len(sentences)):
        candidates.append([' '.join([s for s in sentences[i]])][0])
        references.append([' '.join(l) for l in reference_list[i]][0])
    
    hyp_path ='/tmp/hyp.txt'
    with open(hyp_path, 'w+') as f:
        for line in candidates:
            f.write('{}\n'.format(line))
    
    ref_path = '/tmp/ref.txt'
    with open(ref_path, 'w+') as f:
        for line in references:
            f.write('{}\n'.format(line))
    
    temp_out = StringIO()
    sys.stdout = temp_out
    files2rouge.run(hyp_path, ref_path)
    sys.stdout = sys.__stdout__
    output = temp_out.getvalue()

    rouge_1 = float(re.findall('ROUGE-1 Average_F: (\d*\.?\d+)', output)[0])
    rouge_2 = float(re.findall('ROUGE-2 Average_F: (\d*\.?\d+)', output)[0])
    rouge_l = float(re.findall('ROUGE-L Average_F: (\d*\.?\d+)', output)[0])

    rouge_scores = dict()

    rouge_scores['ROUGE-1'] = [rouge_1]
    rouge_scores['ROUGE-2'] = [rouge_2]
    rouge_scores['ROUGE-L'] = [rouge_l]

    return rouge_scores

def tokenize(s, sub):
    split_toks = s.strip().split()
    tokens = []
    for t in split_toks:
        if re.match(SPECIAL_BLOCK_REGEX,t):
            (prefix, special_token, suffix) = re.findall(SPECIAL_BLOCK_REGEX, t)[0]
            if len(prefix) > 0:
                tokens.append(prefix)
            tokens.append(special_token)
            if len(suffix) > 0:
                tokens.append(suffix)
        else:
            tokens.extend(re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", t))
    if sub:
        return subtokenize(tokens)
    else:
        lowered_tokens = []
        for t in tokens:
            if t not in MARKERS:
                lowered_tokens.append(t.lower())
            else:
                lowered_tokens.append(t)
        return lowered_tokens

def subtokenize(tokens):
    subtokens = []
    for token in tokens:
        if token in MARKERS:
            subtokens.append(token)
            continue

        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        try:
            new_curr = []
            for c in curr:
                by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                new_curr = new_curr + by_symbol

            curr = new_curr
        except:
            curr = []
        if len(curr) == 0:
            continue
        if len(curr) == 1:
            subtokens.append(curr[0].lower())
            continue
        
        for s, subtoken in enumerate(curr):
            subtokens.append(curr[s].lower())
    
    return subtokens

def merge_encoder_outputs(a_states, a_lengths, b_states, b_lengths, device):
    a_max_len = a_states.size(1)
    b_max_len = b_states.size(1)
    combined_len = a_max_len + b_max_len
    padded_b_states = torch.zeros([b_states.size(0), combined_len, b_states.size(-1)], dtype=b_states.dtype, device=device)
    padded_b_states[:, :b_max_len, :] = b_states    
    full_matrix = torch.cat([a_states, padded_b_states], dim=1)
    a_idxs = torch.arange(combined_len, dtype=torch.long, device=device).view(-1, 1)
    b_idxs = torch.arange(combined_len, dtype=torch.long,
                    device=device).view(-1,1) - a_lengths.view(1, -1) + a_max_len
    idxs = torch.where(b_idxs < a_max_len, a_idxs, b_idxs).permute(1, 0)
    offset = torch.arange(0, full_matrix.size(0) * full_matrix.size(1), full_matrix.size(1), device=device)
    idxs = idxs + offset.unsqueeze(1)
    combined_states = full_matrix.reshape(-1, full_matrix.shape[-1])[idxs]
    combined_lengths = a_lengths + b_lengths
    return combined_states, combined_lengths

# Copied from https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py
# Compatibility issues so couldn't import package
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)