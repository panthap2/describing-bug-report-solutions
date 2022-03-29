from einops import rearrange
import itertools
import json
import logging
import os
from argparse import Namespace

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from functools import partial

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    iterators
)
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from multitask_dataset import *

import sys
sys.path.append('../')
from constants import *
from utils import *

EVAL_BLEU_ORDER = 4

def load_multitask_dataset(data_path, split, src, src_dict, tgt, tgt_dict, label_dict, combine, dataset_impl, upsample_primary, left_pad_source,
                          left_pad_target, max_source_positions, max_target_positions, prepend_bos=False, load_alignments=False,
                          truncate_source=False, append_source_id=False, num_buckets=0, shuffle=True, pad_to_multiple=1):
    
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        with open(prefix + 'label') as f:
            lines = f.readlines()  
        label_dataset = [int(l.strip()) for l in lines]

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return MultiTaskDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        label_dataset,
        None,
        label_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task('combined_task')
class CombinedTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--extra-lang-symbol', default='', type=str,
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        
        parser.add_argument('--c_only', action='store_true')

    def __init__(self, args, src_dict, tgt_dict, label_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(",")
        self.extra_langs = args.extra_lang_symbol.split(",")
        for d in [self.src_dict, self.tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            for l in self.extra_langs:
                d.add_symbol("[{}]".format(l))
            d.add_symbol("<mask>")
        
        self.label_dict = label_dict
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        label_dict = cls.load_dictionary(LABEL_DICT)

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict, label_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_multitask_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            self.label_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        
        model.register_classification_head('when2speak', num_classes=2)

        return model
    
    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        assert len(models) == 1
        with torch.no_grad():
            if self.args.c_only:
                gen_out = None
            else:
                gen_out = generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)
           
            if not self.args.c_only:
                data = [gen_out[i][0]["tokens"] for i in range(len(gen_out))]
                max_len = max([x.squeeze().numel() for x in data])
                data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in data]
                hypos = torch.stack(data)
            else:
                hypos = None

            _, _, logits = models[0](src_tokens=sample["net_input"]['src_tokens'], src_lengths=sample["net_input"]['src_lengths'],
                    prev_output_tokens=hypos, ids=sample['id'])
            probs = torch.nn.functional.softmax(logits, dim=-1)
            classification_predictions = torch.argmax(probs, dim=-1)
            pos_probs =  probs[:,1]
          
            return gen_out, classification_predictions, pos_probs

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        if not model.joint_valid:
            # Only want to generate for label=1
            selected_sample = dict()
            idx = sample['label'].tolist().index(1)

            for k, v in sample.items():
                if k in ['nsentences', 'ntokens']:
                    selected_sample[k] = v
                elif k in ['net_input']:
                    selected_sample[k] = dict()
                    for sk, sv in v.items():
                        selected_sample[k][sk] = sv[idx:idx+1]
                else:
                    selected_sample[k] = v[idx:idx+1]

            gen_out, class_out, pos_probs = self.inference_step(generator, [model], selected_sample, prefix_tokens=None)
            hyps, refs = [], []
            for i in range(len(gen_out)):
                hyps.append(decode(gen_out[i][0]["tokens"]))
                refs.append(
                    decode(
                        utils.strip_pad(selected_sample["target"][i], self.tgt_dict.pad()),
                        escape_unk=True,  # don't count <unk> as matches to the hypo
                    )
                )

        else:
            gen_out, class_out, pos_probs = self.inference_step(generator, [model], sample, prefix_tokens=None)

            pos = None
            for j in range(len(sample['id'])):
                # Find the smallest position for which the positive label is predicted
                if class_out[j] == 1 and (pos is None or sample['id'][pos] > sample['id'][j]):
                    pos = j
                        
            hyps, refs = [], []

            if pos is not None:
                hyps.append(decode(gen_out[pos][0]["tokens"]))
                refs.append(
                    decode(
                        utils.strip_pad(sample["target"][pos], self.tgt_dict.pad()),
                        escape_unk=True,  # don't count <unk> as matches to the hypo
                    )
                )
            
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
    
    def get_batch_iterator(self, dataset, max_tokens=None, max_sentences=None, max_positions=None, ignore_invalid_inputs=False, required_batch_size_multiple=1,
                           seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1, data_buffer_size=0, disable_iterator_cache=False, force_generation_turn=True):
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(dataset)
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)
        
        # create mini-batches with given size constraints
        batch_sampler, indices = dataset.batch_by_size(
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            force_generation_turn=force_generation_turn,
            last_sample=True
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            # collate_fn=dataset.collater,
            collate_fn=partial(dataset.collater, pad_to_length={'source': self.max_positions()[0], 'target': None}),
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

# https://github.com/pytorch/fairseq/blob/main/fairseq/models/bart/model.py
@register_model("multi_bart")
class MultiBARTModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

        self.c_only = args.c_only
        self.decoder_state = not self.c_only
        self.prev_decoder_state = not self.c_only
        self.joint_valid = not self.c_only

    @staticmethod
    def add_args(parser):
        super(MultiBARTModel, MultiBARTModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(self, src_tokens, src_lengths, prev_output_tokens, ids, features_only=False, token_embeddings=None, **kwargs):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            **kwargs,
        )

        if not self.c_only:
            x, extra = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                **kwargs,
            )
        else:
            x = None
            extra = None

        encoder_hidden_states = encoder_out.encoder_out # T x B x H
        encoder_hidden_states = rearrange(encoder_hidden_states, 't b h -> b t h')
        sentence_representation = encoder_hidden_states[
            src_tokens.eq(self.encoder.dictionary.eos()), :].view(encoder_hidden_states.size(0), -1, encoder_hidden_states.size(-1))[:, -1, :]
        
        if self.prev_decoder_state or self.decoder_state:
            last_decoder_states = rearrange(extra['inner_states'][-1], 't b h -> b t h')
            
            # Teacher forcing during training
            # Test time, need to take the classification state corresponding to the last decoder step
            decoder_sentence_representation = last_decoder_states[
                prev_output_tokens.eq(self.decoder.dictionary.eos()), :].view(last_decoder_states.size(0), -1, last_decoder_states.size(-1))[:, -1, :]
        
        if self.decoder_state:
            sentence_representation = torch.cat([sentence_representation, decoder_sentence_representation], dim=-1)
        
        if self.prev_decoder_state:
            _, sort_order = ids.sort(descending=False)
            prev_ids = torch.ones_like(ids) * -1

            for i in range(1, len(sort_order)):
                orig_pos = sort_order[i]
                prev_pos = sort_order[i-1]
                prev_ids[orig_pos] = prev_pos
            
            prev_decoder_sentence_representation = torch.zeros_like(decoder_sentence_representation)
            
            for i in range(prev_decoder_sentence_representation.shape[0]):
                prev_pos = prev_ids[i]
                if prev_pos != -1:
                    prev_decoder_sentence_representation[i] = decoder_sentence_representation[prev_pos]
            
            if self.prev_decoder_state:
                sentence_representation = torch.cat([sentence_representation, prev_decoder_sentence_representation], dim=-1)
            
        classification_output = self.classification_heads['when2speak'](sentence_representation)
    
        return x, extra, classification_output

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        input_dim = self.args.encoder_embed_dim
        if self.decoder_state:
            input_dim += self.args.decoder_output_dim
        if self.prev_decoder_state:
            input_dim += self.args.decoder_output_dim

        self.classification_heads[name] = BARTClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=False,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    # logger.info("Overwriting", prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    
@register_model_architecture("multi_bart", "multibart_base")
def multibart_base_architecture(args):
    # mbart_base
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    # bart_base
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)

    # bart_large
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


