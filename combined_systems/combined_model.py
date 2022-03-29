import argparse
import ast
from datetime import datetime
import logging
import math
from itertools import chain
import os
import random
import sys
import sentencepiece as spm
import shutil
import subprocess

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
import torch.nn as nn
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    scoring,
    tasks,
    utils,
)

# https://github.com/pytorch/fairseq/blob/ee833ed49d79da00faa396bd0509782841d3d688/fairseq_cli/train.py#L145

from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from multitask_criterion import *
from combined_plbart_task import *
from preprocess import preprocess

sys.path.append('../')
from constants import *
from utils import *

s = spm.SentencePieceProcessor(model_file=SENTENCE_PIECE_MODEL_PATH)
BATCH_SIZE = 16
UPDATE_FREQ = 16
langs = 'java,python,en_XX'

def main(args):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    criterion.set_label_weights(args.negative_weight, args.positive_weight)
    class_lambda, gen_lambda = get_lambdas(args)
    criterion.set_lambdas(class_lambda, gen_lambda)
    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.batch_size
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_losses = [None]
    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats

def format_input_output(ex):
    text = []
    summary = []
    class_label = []

    utterance_inp_list = []

    tokenized_title = [TITLE_CLS] + ex.title.tokens
    utterance_inp_list.extend(tokenized_title)
    before_code_change_turns = [ex.report] + ex.pre_utterances
    
    for u, utterance in enumerate(before_code_change_turns): 
        utterance_inp_list = utterance_inp_list + [UTTERANCE_CLS] + utterance.tokens
        text.append(utterance_inp_list)
        summary.append(ex.solution_description.tokens)

        if u == len(before_code_change_turns)-1:
            class_label.append(1)
        else:
            class_label.append(0)

    return text, summary, class_label

def build_dataset(examples, partition, global_args):
    print('{}: Starting building {}'.format(datetime.now(), len(examples)))
    src = []
    trg = []
    label = []

    for ex in examples:
        text, summary, class_label = format_input_output(ex)
        src.extend(text)
        trg.extend(summary)
        label.extend(class_label)
    
    assert len(src) == len(trg) and len(src) == len(label)
    print('Total turns: {}'.format(len(src)))

    prefix = os.path.join(global_args.processed_data_dir, '{}'.format(partition))
    src_path = prefix + '.source'

    with open(src_path, 'w+') as f:
        for x in src:
            sent_piece = s.encode(' '.join(x))
            trunc_len = min(len(sent_piece), 1020)
            shortened_sent_piece = sent_piece[-trunc_len:]
            w_x = s.decode(shortened_sent_piece).split()
            f.write('{}\n'.format(' '.join(w_x)))
    
    print('Wrote to {}'.format(src_path))
    
    trg_path = prefix + '.target'
    with open(trg_path, 'w+') as f:
        for x in trg:
            f.write('{}\n'.format(' '.join(x)))
    
    print('Wrote to {}'.format(trg_path))
    
    label_path = prefix + '.label'
    with open(label_path, 'w+') as f:
        for x in label:
            f.write('{}\n'.format(x))

    print('Wrote to {}'.format(label_path))
    
    print('{}: Terminating building {}'.format(datetime.now(), len(examples)))
    return prefix

def transform(prefix):
    arguments = [
        'sh',
        'transform.sh',
        prefix,
        SPM_PATH,
        SENTENCE_PIECE_MODEL_PATH,
        PLBART_CHECKPOINT
    ]
    subprocess.run(arguments, check=True)
    shutil.copy('{}.label'.format(prefix), '{}.spm.label'.format(prefix))

def run_train(train_examples, valid_examples, global_args):
    os.makedirs(global_args.model_dir, exist_ok=True)

    print('{}: Starting preprocessing'.format(datetime.now()))
    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))

    train_prefix = build_dataset(train_examples, 'train', global_args)
    valid_prefix = build_dataset(valid_examples, 'valid', global_args)

    transform(train_prefix)
    transform(valid_prefix)
    
    preprocess([
        '--source-lang={}'.format(SOURCE),
        '--target-lang={}'.format(TARGET),
        '--trainpref={}.spm'.format(train_prefix),
        '--validpref={}.spm'.format(valid_prefix),
        '--destdir={}'.format(global_args.processed_data_dir),
        '--thresholdtgt=0',
        '--thresholdsrc=0',
        '--srcdict={}'.format(PLBART_DICT),
        '--tgtdict={}'.format(PLBART_DICT),
        '--workers=70'
    ])
    
    parser = options.get_training_parser()
    label_weights = get_label_weights(train_examples)
    c_lambda, g_lambda = get_lambdas(global_args)
    input_args = [
        '{}'.format(global_args.processed_data_dir),
        '--langs={}'.format(langs),
        '--task=combined_task',
        '--arch=multibart_base',
        '--layernorm-embedding',
        '--source-lang={}'.format(SOURCE),
        '--target-lang={}'.format(TARGET),
        '--criterion=multitask_criterion',
        '--label-smoothing=0.1',
        '--batch-size={}'.format(BATCH_SIZE),
        '--update-freq={}'.format(UPDATE_FREQ),
        '--max-epoch=30',
        '--optimizer=adam',
        '--adam-eps=1e-06',
        "--adam-betas=(0.9, 0.98)",
        '--lr-scheduler=polynomial_decay',
        '--lr=5e-05',
        '--min-lr=-1',
        '--max-update=100000',
        '--dropout=0.1',
        '--attention-dropout=0.1',
        '--weight-decay=0.0',
        '--seed=1234',
        '--log-format=json',
        '--log-interval=100',
        '--restore-file={}'.format(PLBART_CHECKPOINT),
        '--reset-dataloader',
        '--reset-optimizer',
        '--reset-meters',
        '--reset-lr-scheduler',
        '--no-epoch-checkpoints',
        '--patience=10',
        '--ddp-backend=no_c10d',
        '--save-dir={}'.format(global_args.model_dir),
        '--negative_weight={}'.format(label_weights[0]),
        '--positive_weight={}'.format(label_weights[1]),
        # '--class_lambda={}'.format(c_lambda),
        # '--gen_lambda={}'.format(g_lambda)
    ]

    input_args.append('--warmup-updates=500')
    
    if not global_args.c_only:
        input_args.append('--best-checkpoint-metric=bleu')
        input_args.append('--eval-bleu')
        input_args.append('--eval-bleu-detok=space')
        input_args.append('--eval-tokenized-bleu')
        input_args.append('--eval-bleu-remove-bpe=sentencepiece')
        input_args.append('--eval-bleu-args={"beam": 5}')
        input_args.append('--maximize-best-checkpoint-metric')

    if global_args.c_only:
        input_args.append('--c_only')

    args = options.parse_args_and_arch(parser, input_args, modify_parser=None)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)

def evaluate(args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _evaluate(args, h)
    else:
        return _evaluate(args, sys.stdout)

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

def _evaluate(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    if args.lm_path is not None:
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({args.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
        force_generation_turn=False,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(args, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    predicted_class_labels = []
    predicted_class_probs = []
    ret_order = []

    for sample in progress:
        ret_order.extend(sample["id"].tolist())
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample["target"][:, : args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos, class_out, pos_probs = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )

        predicted_class_labels.extend(class_out.tolist())
        predicted_class_probs.extend(pos_probs.tolist())

        if args.c_only:
            continue

        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        args.remove_bpe,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not args.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if args.print_alignment:
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )

                    if args.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if getattr(args, "retain_iter_history", False):
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    
    if not args.c_only:
        logger.info("NOTE: hypothesis and token scores are output in base 2")
        logger.info(
            "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
                num_sentences,
                gen_timer.n,
                gen_timer.sum,
                num_sentences / gen_timer.sum,
                1.0 / gen_timer.avg,
            )
        )
        if has_target:
            if args.bpe and not args.sacrebleu:
                if args.remove_bpe:
                    logger.warning(
                        "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                    )
                else:
                    logger.warning(
                        "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                    )
            # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
            print(
                "Generate {} with beam={}: {}".format(
                    args.gen_subset, args.beam, scorer.result_string()
                ),
                file=output_file,
            )

    return scorer, predicted_class_labels, predicted_class_probs, ret_order

def post_process(output_dir, partition):
    arguments = [output_dir, partition]
    arguments = ['sh', 'post_process.sh'] + arguments
    subprocess.run(arguments, check=True)

def run_test(test_examples, global_args):
    print('{}: Starting preprocessing'.format(datetime.now()))
    print('Test: {}'.format(len(test_examples)))
    test_prefix = build_dataset(test_examples, 'test', global_args)
    
    transform(test_prefix)
    preprocess([
        '--source-lang={}'.format(SOURCE),
        '--target-lang={}'.format(TARGET),
        '--testpref={}.spm'.format(test_prefix),
        '--destdir={}'.format(global_args.processed_data_dir),
        '--thresholdtgt=0',
        '--thresholdsrc=0',
        '--srcdict={}'.format(PLBART_DICT),
        '--tgtdict={}'.format(PLBART_DICT),
        '--workers=70'
    ])

    parser = options.get_generation_parser()
    input_args = [
        '{}'.format(global_args.processed_data_dir),
        '--path={}/checkpoint_best.pt'.format(global_args.model_dir),
        '--langs={}'.format(langs),
        '--task=combined_task',
        '--gen-subset=test',
        '-t={}'.format(TARGET),
        '-s={}'.format(SOURCE),
        '--scoring=sacrebleu',
        '--remove-bpe=sentencepiece',
        '--max-len-b=200',
        '--beam=5',
        '--batch-size={}'.format(BATCH_SIZE),
        '--results-path={}'.format(global_args.output_dir)
  
    ]

    if global_args.c_only:
        input_args.append('--c_only')
    
    args = options.parse_args_and_arch(parser, input_args)
    _, predicted_class_labels, predicted_class_probs, order = evaluate(args)

    if not global_args.c_only:
        post_process(global_args.output_dir, 'test')

        with open(os.path.join(global_args.output_dir, 'hypotheses.txt')) as f:
            pred_lines = f.readlines()
        
        with open(os.path.join(global_args.output_dir, 'target.txt')) as f:
            trg_lines = f.readlines()

        reordered_pred_lines = ['' for _ in range(len(order))]
        reordered_trg_lines = ['' for _ in range(len(order))]
    
    reordered_class_labels = [0 for _ in range(len(order))]
    reordered_class_probs = [0.0 for _ in range(len(order))]
    
    predictions = []
    references = []

    for o, o_idx in enumerate(order):
        if not global_args.c_only:
            reordered_pred_lines[o_idx] = pred_lines[o]
            reordered_trg_lines[o_idx] = trg_lines[o]
        
        reordered_class_labels[o_idx] = predicted_class_labels[o]
        reordered_class_probs[o_idx] = predicted_class_probs[o]
    
    total_turns = 0.0
    for ex in test_examples:
        total_turns += len([ex.report] + ex.pre_utterances)
    
    if not global_args.c_only:
        pred_lines = reordered_pred_lines
        trg_lines = reordered_trg_lines

        assert len(pred_lines) == total_turns
        assert len(trg_lines) == total_turns
    
    predicted_class_labels = reordered_class_labels
    predicted_class_probs = reordered_class_probs

    assert len(predicted_class_labels) == total_turns
    assert len(predicted_class_probs) == total_turns
    
    e = 0
    all_predictions = []
    selected_indices = []

    for k, ex in enumerate(test_examples):
        turns = [ex.report] + ex.pre_utterances
        selected_idx = -1
        optimal_idx = len(turns)-1

        for t, turn in enumerate(turns):
            pred_class_label = predicted_class_labels[e]
            
            if not global_args.c_only:
                pred_line = pred_lines[e].strip()
                all_predictions.append(pred_line)
                
            if selected_idx == -1 and pred_class_label == 1:
                selected_idx = t
            
            e += 1
        
        selected_indices.append(selected_idx)

    if not global_args.c_only:
        all_prediction_path = os.path.join(global_args.output_dir, 'gen_all_turns.txt')
        with open(all_prediction_path, 'w+') as f:
            for p in all_predictions:
                f.write('{}\n'.format(p))
        print('Wrote {} to {}'.format(len(all_predictions), all_prediction_path))

    class_prediction_path = os.path.join(global_args.output_dir, 'class.txt')
    with open(class_prediction_path, 'w+') as f:
        for c in selected_indices:
            f.write('{}\n'.format(c))
    print('Wrote {} to {}'.format(len(selected_indices), class_prediction_path))
    
    print('{}: Terminating evaluation'.format(datetime.now()))

def get_label_weights(train_examples):
    total = 0.0
    for ex in train_examples:
        total += len(ex.pre_utterances) + 1
    
    pos = len(train_examples)
    pos_weight = total/(2*pos)
    neg_weight = total/(2*(total-pos))

    return [neg_weight, pos_weight]

def get_lambdas(global_args):
    if global_args.c_only:
        return 1.0, 0.0
    else:
        return 0.2, 0.8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--valid_mode', action='store_true')
    parser.add_argument('--processed_data_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--c_only', action='store_true')
    args = parser.parse_args()

    global_args = parser.parse_args()

    print('{}: Loading data'.format(datetime.now()))
    train_examples, valid_examples, test_examples = get_data_splits(args.filtered)

    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))

    if global_args.test_mode:
        run_test(test_examples, global_args)
    elif global_args.valid_mode:
        run_test(valid_examples, global_args)
    else:
        run_train(train_examples, valid_examples, global_args)
    