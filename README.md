# Learning to Describe Solutions for Bug Reports Based on Developer Discussions

**Code and datasets for our paper "Learning to Describe Solutions for Bug Reports Based on Developer Discussions" at Findings of ACL 2022**.

Note that the code and data can only be used for research purposes. Running this code requires `torch==1.4.0+cu92` and `fairseq==0.10.2`. Other libraries may need to be installed as well. This code base borrows code from the [PLBART](https://github.com/wasiahmad/PLBART) and [Fairseq](https://github.com/pytorch/fairseq) repositories.


**Getting Started**
1. Clone the repository.
2. Edit `constants.py` by specifying the `ROOT_DIR`.
3. Install [sentencepiece](https://github.com/google/sentencepiece) and specify the path to `spm_encode` as `SPM_PATH` in `constants.py`.
4. Follow directions [here](https://github.com/wasiahmad/PLBART) to download PLBART-BASE.
5. Download our dataset and saved models from [here](XXX).


**Running Generation Model

