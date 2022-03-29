# Learning to Describe Solutions for Bug Reports Based on Developer Discussions

**Code and datasets for our paper "Learning to Describe Solutions for Bug Reports Based on Developer Discussions" at Findings of ACL 2022**.

Note that the code and data can only be used for research purposes. Running this code requires `torch==1.4.0+cu92` and `fairseq==0.10.2`. Other libraries may need to be installed as well. This code base borrows code from the [PLBART](https://github.com/wasiahmad/PLBART) and [Fairseq](https://github.com/pytorch/fairseq) repositories.


**Getting Started**
1. Clone the repository.
2. Edit `constants.py` by specifying the `ROOT_DIR`.
3. Install [sentencepiece](https://github.com/google/sentencepiece) and specify the path to `spm_encode` as `SPM_PATH` in `constants.py`.
4. Download `sentencepiece.bpe.model` and `dict.txt` from [here](https://github.com/wasiahmad/PLBART/tree/main/sentencepiece) and specify the paths for `SENTENCE_PIECE_MODEL_PATH` and `PLBART_DICT` respectively in `constants.py`.
5. Follow directions [here](https://github.com/wasiahmad/PLBART) to download plbart-base.pt and specify the path for `PLBART_CHECKPOINT` in `constants.py`.
6. Download the dataset and saved models from [here](https://drive.google.com/drive/folders/1pirq1EF7UnXpq33Cir3_Sz3l8jv_2kTB?usp=sharing). 
7. Create a directory for writing processed data, which will be referred to as ``[PROCESSED_DATA_DIR]`` in later steps.
8. Create a directory for writing predicted output, which will be referred to as ``[OUTPUT_DIR]`` in later steps.
9. Create a directory for writing new models, which will be referred to as ``[MODEL_DIR]`` in later steps.

The commands below correspond to runnting training and inference on the full dataset. To use the filtered dataset, simply append the ``--filtered`` flag to any command.

**Running PLBART Generation Model**
1. To run inference on the finetuned PLBART generation model, run the following:
```
python3 plbart.py --test_mode --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=finetuned_plbart_generation/ --output_dir=[OUTPUT_DIR]
```

2. To instead finetune the original PLBART checkpoint, and then run inference, run the following commands:
```
python3 plbart.py --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=[MODEL_DIR]
python3 plbart.py --test_mode --processed_data_dir=[PROCESSED_DATA_DIR] --model_dir=[MODEL_DIR] --output_dir=[OUTPUT_DIR]
```

**Running Transformer Generation Models**
