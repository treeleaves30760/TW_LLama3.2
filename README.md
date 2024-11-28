# TW LLama 3.2

This is a project that eval the finetuned llama 3.2 11b vision instruct model.

## Install

You can create a conda to run this project

```bash
conda create -n TW_LLama3.2 python==3.11.9 -y
conda activate TW_LLama3.2
pip install -r requirements.txt
```

Then you need to create the .env file

```.env
LLAMA_BASE=<Your LLAMA base model path>
LLAMA_1EPOCH=<Your LLAMA 1epoch model path>
LLAMA_3EPOCHS=<Your LLAMA 3epochs model path>
LLAMA_5EPOCHS=<Your LLAMA 5epochs model path>
IMAGE_DIR=<Your Image Folder>
```

## Usage

If we want to see the total generated:
```bash
python llama_eval.py
```

If we want to check the closed word accuracy only:
```bash
python llama_eval_closed_words.py
```

## Result

You can refer to results folder to see the result.