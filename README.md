# Learning to Describe Solutions for Bug Reports Based on Developer Discussions

**Code and datasets for our paper "Learning to Describe Solutions for Bug Reports Based on Developer Discussions" at Findings of ACL 2022**.

If you find this work useful, please consider citing our paper:

```
@inproceedings{PanthaplackelETAL22BugReportDescription,
  author = {Panthaplackel, Sheena and Li, Junyi Jessy and Gligoric, Milos and Mooney, Raymond J.},
  title = {Learning to Describe Solutions for Bug Reports Based on Developer Discussions},
  booktitle = {Findings of ACL (Association for Computational Linguistics)},
  pages = {2935--2952},
  year = {2022},
}
```

Note that the code and data can only be used for research purposes. Running this code requires `torch==1.4.0+cu92` and `fairseq==0.10.2`. Other libraries may need to be installed as well. This code base borrows code from the [PLBART](https://github.com/wasiahmad/PLBART) and [Fairseq](https://github.com/pytorch/fairseq) repositories.


**Getting Started**
1. Clone the repository.
2. Edit `constants.py` by specifying the `ROOT_DIR`.
3. Install [sentencepiece](https://github.com/google/sentencepiece) and specify the path to `spm_encode` as `SPM_PATH` in `constants.py`.
4. Download `sentencepiece.bpe.model` and `dict.txt` from [here](https://github.com/wasiahmad/PLBART/tree/main/sentencepiece) and specify the paths for `SENTENCE_PIECE_MODEL_PATH` and `PLBART_DICT` respectively in `constants.py`.
5. Follow directions [here](https://github.com/wasiahmad/PLBART/blob/main/pretrain/download.sh) to download plbart-base.pt and specify the path for `PLBART_CHECKPOINT` in `constants.py`.
6. Download our dataset and saved models from [here](https://drive.google.com/drive/folders/1pirq1EF7UnXpq33Cir3_Sz3l8jv_2kTB?usp=sharing). Note that the primary dataset is located in `public_bug_report_data`.
7. Create a directory for writing processed data, which will be referred to as ``[PROCESSED_DATA_DIR]`` in later steps.
8. Create a directory for writing predicted output, which will be referred to as ``[OUTPUT_DIR]`` in later steps.
9. Create a directory for writing a new model, which will be referred to as ``[MODEL_DIR]`` in later steps.

The commands below correspond to runnting training and inference on the full dataset. To use the filtered dataset, simply append the ``--filtered`` flag to any command.

**Running PLBART Generation Model**
1. To run inference on the finetuned PLBART generation model, run the following:
```
cd generation_models/
python3 plbart.py --test_mode --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=finetuned_plbart_generation/ --output_dir=[OUTPUT_DIR]
```

2. To instead finetune the original PLBART checkpoint, and then run inference, run the following commands:
```
cd generation_models/
python3 plbart.py --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=[MODEL_DIR]
python3 plbart.py --test_mode --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=[MODEL_DIR] --output_dir=[OUTPUT_DIR]
```

**Running Transformer Generation Models**
1. To train and evaluate a transformer-based seq2seq model (with a pointer network) and then run the following commands:
```
cd generation_models/
python3 transformer_seq2seq.py --model_path=[MODEL_DIR]/model.pkl.gz
python3 transformer_seq2seq.py --test_mode --model_path=[MODEL_DIR]/model.pkl.gz
```
2. To train and evaluate a *hierarchical* transformer-based seq2seq model (with a pointer network) and then run the following commands:
```
cd generation_models/
python3 transformer_seq2seq.py --hierarchical --model_path=[MODEL_DIR]/hier_model.pkl.gz
python3 transformer_seq2seq.py --hierarchical --test_mode --model_path=[MODEL_DIR]/hier_model.pkl.gz
```

**Running Pipelined Combined System**
1. To run inference on the already finetuned system, run the following command:
```
cd combined_systems/
python3 pipelined_model.py --class_model_dir=finetuned_plbart_classification/ --gen_model_dir=finetuned_plbart_generation/ --output_dir=[OUTPUT_DIR] --processed_data_dir=--processed_data_dir=[PROCESSED_DATA_DIR] --test_mode 
```
2. To instead finetune the original PLBART checkpoint, you should first finetune a generation model using Step #2 in the section titled "Running PLBART Generation Model." You can then train the classifier and run inference using the following commands:
```
cd combined_systems/
python3 pipelined_model.py --processed_data_dir=--processed_data_dir=[PROCESSED_DATA_DIR] --class_model_dir=[MODEL_DIR] --gen_model_dir=[PATH TO SAVED GENERATION MODEL]
python3 pipelined_model.py --processed_data_dir=--processed_data_dir=[PROCESSED_DATA_DIR] --class_model_dir=[MODEL_DIR] --gen_model_dir=[PATH TO SAVED GENERATION MODEL] --output_dir=[OUTPUT_DIR] --test_mode
```

**Running Jointly Trained Combined System**
1. To run inference on the already finetuned system, run the following command:
```
cd combined_systems/
python3 jointly_trained_model.py --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=finetuned_plbart_joint/ --output_dir=[OUTPUT_DIR]
```
2. To instead finetune the original PLBART checkpoint and run inference, use the following commands:
```
cd combined_systems/
python3 jointly_trained_model.py --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=[MODEL_DIR]
python3 jointly_trained_model.py --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=[MODEL_DIR] --output_dir=[OUTPUT_DIR] --test_mode
```

**Supplementary Data**

We have provided additional data in the `supplementary_data` directory. In our work, we only consider in-lined code snippets and exclude longer code snippets which are marked with markdown tags. However, this information is included in the raw data at `supplementary_data/bugs/single_code_change` (see the `anonymized_raw_text` field). Next, although we only consider bug-related issue reports and those associated with a single commit message/PR title, we have included the raw data for non-bug reports (`supplementary_data/nonbugs/`) and multiple set of code changes/descriptions (`supplementary_data/bugs/multi_code_change` and `supplementary_data/nonbugs/multi_code_change`). 
