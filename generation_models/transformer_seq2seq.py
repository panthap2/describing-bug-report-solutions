import argparse
from collections import Counter
from datetime import datetime
from dpu_utils.mlutils import Vocabulary
from einops import rearrange
from functools import partial
import math
import numpy as np
import os
import random
import subprocess
import sys
import time
import torch
from torch import nn

sys.path.append('../')
from constants import *
from utils import *

# https://andrewpeng.dev/transformer-pytorch/
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PreprocessedData():
    def __init__(self, src_sequence, stepped_src_sequence, trg_sequence):
        self.src = src_sequence
        self.stepped_src_sequence = stepped_src_sequence
        self.trg = trg_sequence

def format_input_output(ex):
    tokenized_title = [TITLE_CLS] + ex.title.tokens
    flat_input_sequence = tokenized_title
    stepped_input_sequence = [tokenized_title]
    
    before_code_change_turns = [ex.report] + ex.pre_utterances
    for u, utterance in enumerate(before_code_change_turns):
        tokenized_utterance = [UTTERANCE_CLS] + utterance.tokens
        flat_input_sequence.extend(tokenized_utterance)
        stepped_input_sequence.append(tokenized_utterance)
    
    output_sequence = [get_sos()] + ex.solution_description.tokens + [get_eos()]
    return PreprocessedData(flat_input_sequence, stepped_input_sequence, output_sequence)
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, input_data_examples, nl_vocabulary,
                 max_input_length, max_output_length, max_step_length, device):
        self.count = len(examples)

        self.input_token_ids = []
        self.input_lengths = []
        self.output_token_ids = []
        self.output_lengths = []
        self.output_extended_token_ids = []

        self.invalid_copy_positions = []
        self.input_str_reps = []
        self.input_extended_token_ids = []

        self.stepped_input_token_ids = []
        self.stepped_input_lengths = []
        self.num_steps = []

        for e, ex in enumerate(examples):
            if args.hierarchical:
                steps = input_data_examples[e].stepped_src_sequence
                ex_step_list = []
                ex_step_length_list = []
                copy_inputs = []
                set_length = max_input_length
                total_length = 0
                s_count = 0
                for step in steps:
                    step_length = min(len(step), max_step_length)
                    if total_length + step_length > max_input_length:
                        step_length = max_input_length - total_length
                        if step_length <= 0:
                            break

                    step_ids = get_padded_nl_ids(step[:step_length], max_step_length, nl_vocabulary)
                    copy_inputs.extend(step[:step_length])
                    ex_step_list.append(step_ids)
                    ex_step_length_list.append(step_length)
                    total_length += step_length
                    s_count += 1

                self.stepped_input_token_ids.append(ex_step_list)
                self.stepped_input_lengths.append(ex_step_length_list)
                self.num_steps.append(s_count)
                self.input_token_ids.append(None)
                self.input_lengths.append(None)
            else:
                input_sequence = input_data_examples[e].src
                input_length = min(len(input_sequence), max_input_length)
                input_ids = get_padded_nl_ids(input_sequence, max_input_length, nl_vocabulary)
                self.input_token_ids.append(input_ids)
                self.input_lengths.append(input_length)
                self.stepped_input_token_ids.append(None)
                self.stepped_input_lengths.append(None)
                self.num_steps.append(None)
                copy_inputs = input_sequence[:input_length]
                set_length = max_input_length

            input_str_reps = []
            input_extended_token_ids = []
            extra_counter = len(nl_vocabulary)
            max_limit = extra_counter + set_length
            for c in copy_inputs:
                nl_id = get_nl_id(c, nl_vocabulary)
                if is_nl_unk(nl_id, nl_vocabulary) and extra_counter < max_limit:
                    if c in input_str_reps:
                        nl_id = input_extended_token_ids[input_str_reps.index(c)]
                    else:
                        nl_id = extra_counter
                        extra_counter += 1

                input_str_reps.append(c)
                input_extended_token_ids.append(nl_id)
            
            input_extended_token_ids = input_extended_token_ids + [
                get_pad_id(nl_vocabulary) for _ in range(set_length-len(input_extended_token_ids))]
            self.input_extended_token_ids.append(input_extended_token_ids)
            
            output_sequence = input_data_examples[e].trg
            output_ids = get_padded_nl_ids(output_sequence, max_output_length, nl_vocabulary)
            output_length = min(len(output_sequence), max_output_length)
            output_extended_token_ids = get_extended_padded_nl_ids(
                output_sequence, max_output_length, input_extended_token_ids, input_str_reps, nl_vocabulary)
            self.output_token_ids.append(output_ids)
            self.output_lengths.append(output_length)
            self.output_extended_token_ids.append(output_extended_token_ids)
            self.invalid_copy_positions.append(get_invalid_copy_locations(input_str_reps, max_input_length,
                output_sequence, max_output_length))
            self.input_str_reps.append(input_str_reps)

    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        return self.input_token_ids[index],\
            self.input_lengths[index],\
            self.output_token_ids[index],\
            self.output_lengths[index],\
            self.output_extended_token_ids[index],\
            self.invalid_copy_positions[index],\
            self.input_str_reps[index],\
            self.input_extended_token_ids[index],\
            self.stepped_input_token_ids[index],\
            self.stepped_input_lengths[index],\
            self.num_steps[index],\

            # 0:  self.input_token_ids
            # 1:  self.input_lengths
            # 2:  self.output_token_ids
            # 3:  self.output_lengths
            # 4:  self.output_extended_token_ids
            # 5:  self.invalid_copy_positions
            # 6:  self.input_str_reps
            # 7:  self.input_extended_token_ids
            # 8:  self.stepped_input_token_ids
            # 9:  self.stepped_input_lengths
            # 10: self.num_steps

def _init_fn(worker_id, seed):
    np.random.seed(seed)

def collate_fn(batch, nl_vocabulary):
    input_token_ids = []
    input_lengths = []
    output_token_ids = []
    output_lengths = []
    output_extended_token_ids = []
    invalid_copy_positions = []
    input_str_reps = []
    input_extended_token_ids = []
    stepped_input_token_ids = []
    stepped_input_lengths = []
    num_steps = []

    for b in batch:
        input_token_ids.append(b[0])
        input_lengths.append(b[1])
        output_token_ids.append(b[2])
        output_lengths.append(b[3])
        output_extended_token_ids.append(b[4])
        invalid_copy_positions.append(b[5])
        input_str_reps.append(b[6])
        input_extended_token_ids.append(b[7])
        stepped_input_token_ids.append(b[8])
        stepped_input_lengths.append(b[9])
        num_steps.append(b[10])


    if not args.hierarchical:
        return torch.tensor(input_token_ids, dtype=torch.int64),\
            torch.tensor(input_lengths, dtype=torch.int64),\
            torch.tensor(output_token_ids, dtype=torch.int64),\
            torch.tensor(output_lengths, dtype=torch.int64),\
            torch.tensor(output_extended_token_ids, dtype=torch.int64),\
            torch.tensor(invalid_copy_positions, dtype=torch.bool),\
            input_str_reps,\
            torch.tensor(input_extended_token_ids, dtype=torch.int64), None, None, None
    else:
        batch_size = len(batch)
        max_len = len(stepped_input_token_ids[0][0])
        max_steps = max(num_steps)

        batch_step_token_ids = torch.full([batch_size, max_steps, max_len], get_pad_id(nl_vocabulary), dtype=torch.int64)
        batch_step_lengths = torch.zeros([batch_size, max_steps], dtype=torch.int64)
        batch_num_steps = torch.tensor(num_steps)

        for b_idx in range(batch_size):
            ex_steps = num_steps[b_idx]
            batch_step_token_ids[b_idx,:ex_steps,:] = torch.tensor(stepped_input_token_ids[b_idx])
            batch_step_lengths[b_idx,:ex_steps] = torch.tensor(stepped_input_lengths[b_idx])
        
        return None, None,\
            torch.tensor(output_token_ids, dtype=torch.int64),\
            torch.tensor(output_lengths, dtype=torch.int64),\
            torch.tensor(output_extended_token_ids, dtype=torch.int64),\
            torch.tensor(invalid_copy_positions, dtype=torch.bool),\
            input_str_reps,\
            torch.tensor(input_extended_token_ids, dtype=torch.int64),\
            batch_step_token_ids, batch_step_lengths, batch_num_steps
             
class Incrementer:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1

class GeneralEncoderDecoder(nn.Module):
    def __init__(self, model_path, max_input_length, max_output_length, nl_vocabulary,
                 hierarchical, max_step_length):
        super(GeneralEncoderDecoder, self).__init__()
        self.model_path = model_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.nl_vocabulary = nl_vocabulary
        self.hierarchical = hierarchical
        self.max_step_length = max_step_length
        self.torch_device_name = 'cpu'

        self.d_model = 64
        self.nhead = 4
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.dim_feedforward = 256
        self.lr = 3e-05
        self.batch_size = 8
        self.dropout_rate = 0.2
        self.positional_embedding_size = 8

        curr_time = time.time()
        print('Current time: {}'.format(curr_time))
        self.seed = int(curr_time)
        
        print('Seed: {}'.format(self.seed))
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def get_vocab_size(self):
        return len(self.nl_vocabulary)
    
    def get_nl_embeddings(self, token_ids, decoding=False, attn_mask=None):
        rearranged_input = rearrange(token_ids, 'n t -> t n')
        return self.pos_enc(self.embeddings(rearranged_input) * math.sqrt(self.d_model)) 
    
    def initialize(self):
        self.embeddings = nn.Embedding(self.get_vocab_size(), self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, self.dropout_rate, max(self.max_input_length, self.max_output_length))
        self.transformer = nn.Transformer(self.d_model, self.nhead, self.num_encoder_layers, self.num_decoder_layers,
            self.dim_feedforward, self.dropout_rate)
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.model_output_size = self.d_model
      
        synthesis_size = self.d_model
        if self.hierarchical:
            self.context_encoder = nn.GRU(input_size=self.model_output_size, hidden_size=self.d_model,
                dropout=self.dropout_rate, num_layers=NUM_LAYERS, batch_first=True, bidirectional=False)
            synthesis_size += self.d_model
            self.context_to_decoder_initial = nn.Parameter(torch.randn(self.d_model,
                self.model_output_size, dtype=torch.float, requires_grad=True))

        self.generation_output_matrix = nn.Parameter(
            torch.randn(self.model_output_size, self.get_vocab_size(),
                dtype=torch.float, requires_grad=True)
            )
        
        self.copy_encoder_hidden_transform_matrix = nn.Parameter(
            torch.randn(self.model_output_size, self.model_output_size,
                dtype=torch.float, requires_grad=True)
            )

        if self.hierarchical:
            self.context_synthesis_layer = nn.Linear(synthesis_size, self.d_model, bias=False)

        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.num_epochs = 0
        self.best_loss = float('inf')
        self.patience_tally = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        print('Initialization complete')
        print('Max input length: {}'.format(self.max_input_length))
        print('Max output length: {}'.format(self.max_output_length))
        print('NL vocab size: {}'.format(self.get_vocab_size()))
        print(self)
        sys.stdout.flush()
    
    def synthesize(self, src, context_state, device):
        if not self.hierarchical:
            return src

        synthesis_input = src
        if self.hierarchical:
            if context_state is not None:
                tiled_context_state = context_state.unsqueeze(0).expand(src.shape[0], -1, -1)
            else:
                tiled_context_state = torch.zeros([src.shape[0], src.shape[1], self.d_model], dtype=torch.float32, device=device)
            synthesis_input = torch.cat([synthesis_input, tiled_context_state], dim=-1)
        
        return self.context_synthesis_layer(synthesis_input)

    def get_encoder_output(self, input_token_ids, input_lengths, device, context_state=None):
        src_key_padding_mask = (torch.arange(
            input_token_ids.shape[1], device=device).view(1, -1) >= input_lengths.to(device).view(-1, 1))
        memory_key_padding_mask = src_key_padding_mask.clone()
        src = self.get_nl_embeddings(input_token_ids.to(device), attn_mask=(~memory_key_padding_mask).type(torch.float32))      

        src = self.synthesize(src, context_state, device)
        encoder_hidden_states = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        encoder_final_state = encoder_hidden_states[0,:,:]
        encoder_hidden_states = rearrange(encoder_hidden_states, 's n e -> n s e')
        return encoder_hidden_states, memory_key_padding_mask, encoder_final_state
    
    def get_decoder_output(self, encoder_hidden_states, memory_key_padding_mask, output_token_ids, output_lengths, initial_state, device):
        tgt = self.get_nl_embeddings(output_token_ids.to(device), decoding=True)
        if output_lengths is not None:
            tgt_key_padding_mask = (torch.arange(
                tgt.shape[0], device=device).view(1, -1) >= output_lengths.to(device).view(-1, 1))
        else:
            tgt_key_padding_mask = None
        
        tgt_mask = gen_nopeek_mask(tgt.shape[0]).to(device)
        reshaped_encoder_hidden_states = rearrange(encoder_hidden_states, 'n s e -> s n e')
        decoder_hidden_states = self.decoder(tgt, reshaped_encoder_hidden_states, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        decoder_hidden_states = rearrange(decoder_hidden_states, 't n e -> n t e')
        decoder_final_state = None
      
        generation_scores = torch.einsum('ijk,km->ijm', decoder_hidden_states, self.generation_output_matrix)
        logits = generation_scores
        
        copy_scores = torch.einsum('ijk,km,inm->inj', encoder_hidden_states,
            self.copy_encoder_hidden_transform_matrix, decoder_hidden_states)
        copy_scores.masked_fill_(memory_key_padding_mask.unsqueeze(1), float('-inf'))
        logits = torch.cat([logits, copy_scores], dim=-1)
        
        combined_logprobs = nn.functional.log_softmax(logits, dim=-1)
        generation_logprobs = combined_logprobs[:,:,:self.get_vocab_size()]

        copy_logprobs = combined_logprobs[:, :,self.get_vocab_size():]

        return generation_logprobs, copy_logprobs, decoder_final_state
    
    def get_stepped_encoder_output(self, stepped_input_token_ids, stepped_input_lengths, num_steps, device):
        max_steps = torch.max(num_steps)
        encoder_hidden_states = None
        memory_key_padding_mask = None
        encoder_lengths = None
        context_state = None
        for s in range(max_steps):
            min_lengths = torch.ones_like(stepped_input_lengths[:,s])
            lengths = torch.where(s < num_steps, stepped_input_lengths[:,s], min_lengths)
  
            step_hidden_states, step_memory_key_padding_mask, step_final_state = self.get_encoder_output(
                stepped_input_token_ids[:,s], lengths, device, context_state)
            
            prev_context_state = None
            if context_state is not None:
                prev_context_state = context_state.unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
            self.context_encoder.flatten_parameters()
            new_context_state, _ = self.context_encoder(step_final_state.unsqueeze(1), prev_context_state)

            is_step = (s < num_steps).unsqueeze(-1)
            if context_state is not None:
                context_state = torch.where(is_step.to(device), new_context_state.squeeze(1), context_state)
            else:
                context_state = new_context_state.squeeze(1)

            if encoder_hidden_states is None:
                encoder_hidden_states = step_hidden_states
                encoder_lengths = stepped_input_lengths[:,s].to(device)
            else:
                encoder_hidden_states, encoder_lengths = merge_encoder_outputs(
                    encoder_hidden_states, encoder_lengths, step_hidden_states, stepped_input_lengths[:,s].to(device), device)
        
        empty_encoder_hidden_states = torch.zeros([encoder_hidden_states.shape[0], self.max_input_length, encoder_hidden_states.shape[-1]], dtype=torch.float32, device=device)
        min_len = min(encoder_hidden_states.shape[1], self.max_input_length)
        empty_encoder_hidden_states[:,:min_len,:] = encoder_hidden_states[:,:min_len,:]
        encoder_hidden_states = empty_encoder_hidden_states
        memory_key_padding_mask = (torch.arange(
            encoder_hidden_states.shape[1], device=device).view(1, -1) >= encoder_lengths.to(device).view(-1, 1))
        encoder_final_state = torch.einsum('bd,dh->bh', context_state.squeeze(1), self.context_to_decoder_initial)

        return encoder_hidden_states, memory_key_padding_mask, encoder_final_state 
    
    def forward(self, input_token_ids, input_lengths, output_token_ids, output_lengths, output_extended_token_ids,
                invalid_copy_positions, input_str_reps, input_extended_token_ids, stepped_input_token_ids, stepped_input_lengths,
                num_steps):
        try:
            device = torch.cuda.current_device()
        except:
            device = 'cpu'
        
        if self.hierarchical:
            encoder_hidden_states, memory_key_padding_mask, encoder_final_state = self.get_stepped_encoder_output(stepped_input_token_ids,
                stepped_input_lengths, num_steps, device)
        else:
            encoder_hidden_states, memory_key_padding_mask, encoder_final_state = self.get_encoder_output(input_token_ids, input_lengths, device)
        
        generation_logprobs, copy_logprobs, decoder_final_state = self.get_decoder_output(
            encoder_hidden_states, memory_key_padding_mask, output_token_ids[:,:-1], output_lengths, encoder_final_state, device)
        
        gold_generation_ids = output_token_ids[:, 1:].unsqueeze(-1).to(device)
        gold_generation_logprobs = torch.gather(input=generation_logprobs, dim=-1,
                                                index=gold_generation_ids).squeeze(-1)
        
        gold_logprobs = gold_generation_logprobs.unsqueeze(-1)

        copy_logprobs = copy_logprobs.masked_fill(
            invalid_copy_positions.to(device)[:,1:,:encoder_hidden_states.shape[1]], float('-inf'))
        gold_copy_logprobs = copy_logprobs.logsumexp(dim=-1)
        gold_logprobs = torch.cat([gold_logprobs, gold_copy_logprobs.unsqueeze(-1)], dim=-1)
        
        gold_logprobs = torch.logsumexp(gold_logprobs, dim=-1)
        gold_logprobs = gold_logprobs.masked_fill(torch.arange(output_token_ids[:,1:].shape[-1],
            device=device).unsqueeze(0) >= output_lengths.to(device).unsqueeze(-1)-1, 0)
        likelihood_by_example = gold_logprobs.sum(dim=-1)

        # Normalizing by length. Seems to help
        likelihood_by_example = likelihood_by_example/(output_lengths.to(device)-1).float()
        return -(likelihood_by_example).mean()

    def run_inference(self, input_token_ids, input_lengths, input_str_reps, input_extended_token_ids, stepped_input_token_ids,
                      stepped_input_lengths, num_steps):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device('cpu')

        if self.hierarchical:
            encoder_hidden_states, memory_key_padding_mask, encoder_final_state = self.get_stepped_encoder_output(stepped_input_token_ids,
                stepped_input_lengths, num_steps, device)
        else:
            encoder_hidden_states, memory_key_padding_mask, encoder_final_state = self.get_encoder_output(input_token_ids, input_lengths, device)

        predicted_tokens = self.beam_decode(encoder_hidden_states, memory_key_padding_mask, input_str_reps,
            input_extended_token_ids, encoder_final_state, device)

        return predicted_tokens
    
    def beam_decode(self, encoder_hidden_states, memory_key_padding_mask, input_str_reps, input_extended_token_ids, encoder_final_state, device):
        """Beam search. Generates the top K candidate predictions."""
        batch_size = memory_key_padding_mask.shape[0]

        decoder_input = torch.tensor(
            [[get_nl_id(get_sos(), self.nl_vocabulary)]] * batch_size,
            device=self.get_device()
        )
        decoder_input = decoder_input.unsqueeze(1) # [batch_size, curr_len, beam_size]

        beam_scores = torch.ones([batch_size, 1], dtype=torch.float32, device=device)
        beam_status = torch.zeros([batch_size, 1], dtype=torch.uint8, device=device)
        beam_predicted_ids = torch.full([batch_size, 1, self.max_output_length], get_nl_id(get_eos(), self.nl_vocabulary),
            dtype=torch.int64, device=device)
        beam_predicted_ids[:,:,0] = get_nl_id(get_sos(), self.nl_vocabulary)
        decoder_state = encoder_final_state.unsqueeze(1).expand(
            -1, decoder_input.shape[1], -1).reshape(-1, encoder_final_state.shape[-1])

        for i in range(1, self.max_output_length):
            beam_size = decoder_input.shape[1]

            if beam_status[:,0].sum() == batch_size:
                break

            tiled_encoder_states = encoder_hidden_states.unsqueeze(1).expand(-1, beam_size, -1, -1) # [batch_size, beam_size, L, H]
            tiled_memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).expand(-1, beam_size, -1) # [batch_size, beam_size, L]

            flat_decoder_input = decoder_input.reshape(-1, decoder_input.shape[-1]) # [batch_size*beam_size, N]
            flat_encoder_states = tiled_encoder_states.reshape(-1, tiled_encoder_states.shape[-2], tiled_encoder_states.shape[-1]) # [batch_size*beam_size, L, H]
            flat_memory_key_padding_mask = tiled_memory_key_padding_mask.reshape(-1, tiled_memory_key_padding_mask.shape[-1]) # [batch_size*beam_size, L]

            generation_logprobs, copy_logprobs, flat_decoder_state = self.get_decoder_output(
                flat_encoder_states, flat_memory_key_padding_mask, flat_decoder_input, output_lengths=None, initial_state=decoder_state, device=device)
            generation_logprobs = generation_logprobs[:,-1,:]
            generation_logprobs = generation_logprobs.reshape(batch_size, beam_size, generation_logprobs.shape[-1]) # [batch_size, beam_size, V]
            num_scores = generation_logprobs.shape[-1]
            
            copy_logprobs = copy_logprobs[:,-1,:]
            num_scores += copy_logprobs.shape[-1]
            copy_logprobs = copy_logprobs.reshape(batch_size, beam_size, copy_logprobs.shape[-1]) # [batch_size, beam_size, L]
        
            prob_scores = torch.zeros([batch_size, beam_size, num_scores], dtype=torch.float32, device=self.get_device()) # [batch_size, beam_size, V+L]
            prob_scores[:, :, :generation_logprobs.shape[-1]] = torch.exp(generation_logprobs)

            # Factoring in the copy scores
            expanded_token_ids = input_extended_token_ids.unsqueeze(1).expand(-1, beam_size, -1) # [batch_size, beam_size, L]
            prob_scores += scatter_add(src=torch.exp(copy_logprobs), index=expanded_token_ids.to(device), out=torch.zeros_like(prob_scores))
    
            if i >= 1:
                # prev_ids = beam_predicted_ids[:,:,:i] # This might be too restrictive
                if i >= 3:
                    prev_ids = beam_predicted_ids[:,:,i-3:i]
                elif i >= 2:
                    prev_ids = beam_predicted_ids[:,:,i-2:i]
                else:
                    prev_ids = beam_predicted_ids[:,:,i-1:i]

                batch_dim = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1)
                beam_dim = torch.arange(beam_size).unsqueeze(0).unsqueeze(-1)
                prob_scores[batch_dim,beam_dim,prev_ids] = 0.0

            # Extending each of the k beams to the next k beams (total = k*M)
            top_scores_per_beam, top_indices_per_beam = torch.topk(prob_scores, k=BEAM_SIZE, dim=-1) # [batch_size, beam_size, M]
            updated_scores = torch.einsum('eb,ebm->ebm', beam_scores, top_scores_per_beam) # [batch_size, beam_size, M]
            retained_scores = beam_scores.unsqueeze(-1).expand(-1, -1, top_scores_per_beam.shape[-1]) # [batch_size, beam_size, M]

            # Trying to keep at most one ray corresponding to completed beams
            end_mask = (torch.arange(beam_size) == 0).type(torch.float32).to(device)
            end_scores = torch.einsum('b,ebm->ebm', end_mask, retained_scores)

            possible_next_scores = torch.where(beam_status.unsqueeze(-1) == 1, end_scores, updated_scores) # [batch_size, beam_size, M]
            possible_next_status = torch.where(top_indices_per_beam == get_nl_id(get_eos(), self.nl_vocabulary),
                torch.ones([batch_size, beam_size, top_scores_per_beam.shape[-1]], dtype=torch.uint8, device=device),
                beam_status.unsqueeze(-1).expand(-1,-1,top_scores_per_beam.shape[-1])) # [batch_size, beam_size, M]

            possible_beam_predicted_ids = beam_predicted_ids.unsqueeze(2).expand(-1, -1, top_scores_per_beam.shape[-1], -1) # [batch_size, beam_size, M, N]
            
            pool_next_scores = possible_next_scores.reshape(batch_size, -1) # [batch_size, beam_size*M]
            pool_next_status = possible_next_status.reshape(batch_size, -1) # [batch_size, beam_size*M]
            pool_next_ids = top_indices_per_beam.reshape(batch_size, -1) # [batch_size, beam_size*M]
            pool_predicted_ids = possible_beam_predicted_ids.reshape(batch_size, -1, beam_predicted_ids.shape[-1]) # [batch_size, beam_size*M, N]

            # Selecting top k from k*M
            top_scores, top_indices = torch.topk(pool_next_scores, k=BEAM_SIZE, dim=-1) # [batch_size, BEAM_SIZE]
            next_step_ids = torch.gather(pool_next_ids, -1, top_indices) # [batch_size, BEAM_SIZE]
            beam_status = torch.gather(pool_next_status, -1, top_indices) # [batch_size, BEAM_SIZE]
            beam_scores = torch.gather(pool_next_scores, -1, top_indices) # [batch_size, BEAM_SIZE]

            end_tags = torch.full_like(next_step_ids, get_nl_id(get_eos(), self.nl_vocabulary)) # [batch_size, BEAM_SIZE]
            next_step_ids = torch.where(beam_status == 1, end_tags, next_step_ids) # [batch_size, BEAM_SIZE]

            beam_predicted_ids = torch.gather(pool_predicted_ids, 1, top_indices.unsqueeze(-1).expand(-1, -1, pool_predicted_ids.shape[-1])) # [batch_size, BEAM_SIZE, N]
            beam_predicted_ids[:,:,i] = next_step_ids # [batch_size, BEAM_SIZE, N]

            raw_next_input = beam_predicted_ids[:,:,:i+1]
            unks = torch.full_like(raw_next_input, get_nl_id(Vocabulary.get_unk(), self.nl_vocabulary)) # [batch_size, BEAM_SIZE]
            decoder_input = torch.where(raw_next_input < self.get_vocab_size(), raw_next_input, unks) # [batch_size, BEAM_SIZE, N]

        decoded_tokens = []
        for i in range(batch_size):
            token_ids = beam_predicted_ids[i][0][1:].cpu() # First token was SOS, placed there automatically
            tokens = get_nl_tokens(token_ids, list(input_extended_token_ids[i].cpu()), input_str_reps[i], self.nl_vocabulary)
            decoded_tokens.append(tokens)
        return decoded_tokens

    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def get_batch_ids(dataset, batch_size, shuffle=False):
    all_ids = list(range(len(dataset)))
    if shuffle:
        random.shuffle(all_ids)
    
    batches = []
    start_idx = 0
    while start_idx < len(all_ids):
        end_idx = min(start_idx + batch_size, len(all_ids))
        examples = all_ids[start_idx:end_idx]
        if len(examples) > 0:
            batches.append(examples)
        start_idx = end_idx

    return batches

def get_batches(dataset, batch_size, shuffle=False):
    batches = []
    start_idx = 0
    while start_idx < len(dataset):
        end_idx = min(start_idx + batch_size, len(dataset))
        examples = dataset[start_idx:end_idx]
        if len(examples) > 0:
            batches.append(examples)
        start_idx = end_idx

    return batches

def get_sos():
    return SOS

def get_eos():
    return EOS

def get_pad():
    return Vocabulary.get_pad()

def get_unk():
    return Vocabulary.get_unk()

def get_padded_nl_ids(nl_sequence, pad_length, nl_vocabulary):
    return nl_vocabulary.get_id_or_unk_multiple(nl_sequence,
        pad_to_size=pad_length, padding_element=nl_vocabulary.get_id_or_unk(Vocabulary.get_pad()))

def get_nl_id(token, nl_vocabulary):
    return nl_vocabulary.get_id_or_unk(token)

def is_nl_unk(id, nl_vocabulary):
    return id == nl_vocabulary.get_id_or_unk(Vocabulary.get_unk())

def get_pad_id(nl_vocabulary):
    return nl_vocabulary.get_id_or_unk(Vocabulary.get_pad())

def get_extended_padded_nl_ids(nl_sequence, pad_length, inp_ids, inp_tokens, nl_vocabulary):
    # Derived from: https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/mlutils/vocabulary.py
    nl_ids = []
    for token in nl_sequence:
        nl_id = get_nl_id(token, nl_vocabulary)
        if is_nl_unk(nl_id, nl_vocabulary) and token in inp_tokens:
            copy_idx = inp_tokens.index(token)
            nl_id = inp_ids[copy_idx]
        nl_ids.append(nl_id)
    
    if len(nl_ids) > pad_length:
        return nl_ids[:pad_length]
    else:
        padding = [get_pad_id(nl_vocabulary)] * (pad_length - len(nl_ids))
        return nl_ids + padding

def get_vocab_extended_nl_token(token_id, inp_ids, inp_tokens, nl_vocabulary):
    if token_id < len(nl_vocabulary):
        return get_nl_token(token_id, nl_vocabulary)
    elif token_id in inp_ids:
        copy_idx = inp_ids.index(token_id)
        return inp_tokens[copy_idx]
    else:
        return get_unk()

def get_nl_tokens(token_ids, inp_ids, inp_tokens, nl_vocabulary):
    tokens = [get_vocab_extended_nl_token(t, inp_ids, inp_tokens, nl_vocabulary) for t in token_ids]
    eos = get_eos()
    pad = get_pad()
    if eos in tokens:
        tokens = tokens[:tokens.index(eos)]
    if pad in tokens:
        tokens = tokens[:tokens.index(pad)]

    return tokens

def get_nl_token(token_id, nl_vocabulary):
    return nl_vocabulary.get_name_for_id(token_id)

def format_sequence(tokens):
    formatted_tokens = []
    seen_tokens = set()
    last_token = None
    
    for t in tokens:
        if t in [UTTERANCE_CLS, SOS, EOS]:
            continue
        
        if t in MARKERS:
            continue
        
        if t == last_token:
            continue
        
        last_token = t
        seen_tokens.add(t)
        formatted_tokens.append(t)

    return formatted_tokens

def get_invalid_copy_locations(input_sequence, max_input_length, output_sequence, max_output_length):
    input_length = min(len(input_sequence), max_input_length)
    output_length = min(len(output_sequence), max_output_length)

    invalid_copy_locations = np.ones([max_output_length, max_input_length], dtype=np.bool)
    for o in range(output_length):
        for i in range(input_length):
            invalid_copy_locations[o,i] = output_sequence[o] != input_sequence[i]

    return invalid_copy_locations
    
def test(test_data):
    print('Loading model from {}'.format(args.model_path))
    if torch.cuda.is_available():
        model = torch.load(args.model_path)
    else:
        model = torch.load(args.model_path, map_location=torch.device('cpu'))    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.torch_device_name = 'gpu'
        model.cuda()
        for c in model.children():
            c.cuda()
    else:
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
    
    model.eval()

    test_input_data = [format_input_output(ex) for ex in test_data]
    dataset = Dataset(test_data, test_input_data, model.nl_vocabulary,
        model.max_input_length, model.max_output_length, model.max_step_length, device)

    test_batch_generator = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size,
        shuffle=False, collate_fn=partial(collate_fn, nl_vocabulary=model.nl_vocabulary))

    # Lists of string predictions
    gold_strs = []
    pred_strs = []
    references = []
    predictions = []
    abstractive_sentences = []

    with torch.no_grad():
        for batch_data in test_batch_generator:
            sys.stdout.flush()

            b_tokens = model.run_inference(
                batch_data[0], batch_data[1], batch_data[6], batch_data[7], batch_data[8], batch_data[9], batch_data[10])
            abstractive_sentences.extend(b_tokens)
            
    for i in range(len(dataset)):
        gold_obj = test_data[i].solution_description
        predicted_tokens = abstractive_sentences[i]
        
        prediction = format_sequence(predicted_tokens)

        gold_str = ' '.join(gold_obj.tokens)
        references.append([gold_obj.tokens])
        
        pred_str = ' '.join(prediction)
        predictions.append(prediction)
        
        gold_strs.append(gold_str)
        pred_strs.append(pred_str)

        print(test_data[i].issue_url)
        print('Title: {}'.format(' '.join(test_data[i].title.tokens)))
        print('Gold: {}'.format(gold_str))
        print('Prediction: {}'.format(pred_str))
        print('----------------------------')

    scores = compute_scores(references, predictions)
    for metric, vals in scores.items():
        print('{}: {}'.format(metric, 100*sum(vals)/float(len(vals))))
    print('--------------------------------')

def train(train_data, valid_data):
    # Compute stats
    input_lengths = []
    output_lengths = []
    nl_token_counter = Counter()
    step_lengths = []

    for ex in train_data:
        input_data = format_input_output(ex)
        input_lengths.append(len(input_data.src))
        nl_token_counter.update(input_data.src)
        output_lengths.append(len(input_data.trg))
        nl_token_counter.update(input_data.trg)

        if args.hierarchical:
            for step in input_data.stepped_src_sequence:
                step_lengths.append(len(step))

    max_input_length = int(np.percentile(np.asarray(sorted(input_lengths)),
        LENGTH_CUTOFF_PCT))
    max_output_length = int(np.percentile(np.asarray(sorted(output_lengths)),
        LENGTH_CUTOFF_PCT))
    
    max_step_length = 0
    if args.hierarchical:
        max_step_length = int(np.percentile(np.asarray(sorted(step_lengths)),
            LENGTH_CUTOFF_PCT))
        print('Max step length: {}'.format(max_step_length))

    nl_counts = np.asarray(sorted(nl_token_counter.values()))
    nl_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1
    nl_vocabulary = Vocabulary.create_vocabulary(tokens=nl_token_counter,
        max_size=MAX_VOCAB_SIZE, count_threshold=nl_threshold, add_pad=True)

    # Initializing model
    model = GeneralEncoderDecoder(args.model_path, max_input_length, max_output_length, nl_vocabulary, args.hierarchical, max_step_length)
    model.initialize()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.torch_device_name = 'gpu'
        model.cuda()
        for c in model.children():
            c.cuda()
    else:
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
    optimizer = model.optimizer

    train_input_data = [format_input_output(ex) for ex in train_data]
    valid_input_data = [format_input_output(ex) for ex in valid_data]

    while model.num_epochs < 200:
        epoch = model.num_epochs
        if model.patience_tally > PATIENCE:
            print('Terminating')
            break
        
        for param_group in optimizer.param_groups:
            print('{}: Epoch {}, Learning rate: {}'.format(datetime.now(), epoch, param_group['lr']))
            sys.stdout.flush()
        
        model.train()
        num_train_batches = 0
        train_batch_ids = get_batch_ids(train_data, model.batch_size, True)

        train_loss = 0
        for batch_num, batch_ids in enumerate(train_batch_ids):
            print('{}: Epoch {}, Train batch: {}'.format(datetime.now(), epoch, batch_num))
            sys.stdout.flush()
            train_batch_input_data = []
            train_batch_examples = []
            for pos in batch_ids:
                train_batch_input_data.append(train_input_data[pos])
                train_batch_examples.append(train_data[pos])

            train_batch_set = Dataset(train_batch_examples, train_batch_input_data, model.nl_vocabulary,
                model.max_input_length, model.max_output_length, model.max_step_length, device)
            train_batch_generator = torch.utils.data.DataLoader(train_batch_set, batch_size=model.batch_size,
                shuffle=True, worker_init_fn=partial(_init_fn, seed=model.seed),
                collate_fn=partial(collate_fn, nl_vocabulary=model.nl_vocabulary),
                drop_last=True)
             
            for batch_data in train_batch_generator:
                optimizer.zero_grad()
                loss = model.forward(*batch_data).mean()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.cpu())
                num_train_batches += 1
    
        model.eval()
        validation_loss = 0
        num_valid_batches = 0
        valid_batch_ids = get_batch_ids(valid_data, model.batch_size, False)
        valid_predictions = []
        
        with torch.no_grad():
            for batch_num, batch_ids in enumerate(valid_batch_ids):
                print('{}: Epoch {}, Valid batch: {}'.format(datetime.now(), epoch, batch_num))
                sys.stdout.flush()

                valid_batch_input_data = []
                valid_batch_examples = []
                for pos in batch_ids:
                    valid_batch_input_data.append(valid_input_data[pos])
                    valid_batch_examples.append(valid_data[pos])
                
                valid_batch_set = Dataset(valid_batch_examples, valid_batch_input_data, model.nl_vocabulary,
                    model.max_input_length, model.max_output_length, model.max_step_length, device)
                
                valid_batches = get_batches(valid_batch_set, model.batch_size, False)
                
                valid_batch_generator = torch.utils.data.DataLoader(valid_batch_set, batch_size=model.batch_size,
                    shuffle=False, collate_fn=partial(collate_fn, nl_vocabulary=model.nl_vocabulary))

                for batch_data in valid_batch_generator:
                    validation_loss += float(model.forward(*batch_data).mean())
                    num_valid_batches += 1
                
        validation_loss = validation_loss/num_valid_batches
        if validation_loss <= model.best_loss:
            to_save = True
            model.best_loss = validation_loss
            model.patience_tally = 0
        else:
            to_save = False
            model.patience_tally += 1

        model.num_epochs += 1
        model.lr = model.scheduler.get_last_lr()[0]
        print('Epoch: {}'.format(epoch))
        print('Training loss: {}'.format(train_loss/num_train_batches))
        print('Validation loss: {}'.format(validation_loss))
        print('Learning rate: {}'.format(model.lr))

        if to_save:
            torch.save(model, model.model_path)
            print('Saved: {}'.format(model.model_path))
        else:
            print('Decreasing lr')
            model.scheduler.step()
        print('-----------------------------------')
        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchical', action='store_true')
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--valid_mode', action='store_true')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    print('{}: Loading data'.format(datetime.now()))
    train_examples, valid_examples, test_examples = get_data_splits(args.filtered)
    
    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))

    if args.test_mode:
        test(test_examples)
    elif args.valid_mode:
        test(valid_examples)
    else:
        train(train_examples, valid_examples)