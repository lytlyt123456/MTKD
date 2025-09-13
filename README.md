# MTKD: Multi-level Two-stage few-shot Knowledge Distillation

### Preliminary

1. (1) Pre-train your own large teacher CLIP model (See below) or (2) use our publicly released pre-trained teacher ViT-L/14 CLIP models. (**Highly Recommended**)   
Our pre-trained teacher models are publicly available at [[Baidu Yun](https://pan.baidu.com/s/1KNJ1mhNKoxdSli4ZldeZUg?pwd=mjf4)] [[TeraBox](https://terabox.com/s/1X4mxJtSaR8W2lrK5bsrCkg)] [[Google Cloud](https://drive.google.com/drive/folders/1OdQ9WauZmYAzVSUTTw7tIKKChyECIS5B?usp=sharing)]   
(Note that due to cloud space limitations, we only provide a limited number of models in Google Cloud. Sorry.)  
After obtaining the teacher model, unzip these files and place the model in the `./teacher_model` folder.   

2. Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

2. Prepare the dataset. Please follow the instructions detailed in [DATASETS.md](datasets/DATASETS.md). After downloading the datasets, please make a folder named `DATA` and put all the datasets in this folder. For your download convenience, we maintain a repository at huggingface, which contains all the datasets to be used (except imagenet because it is too large).   [[HuggingFace_Download_Links](https://huggingface.co/zhengli97/prompt_learning_dataset)]

### Train Your Teacher Model (Optional)

In our paper, we default use PromptSRC to pre-train our ViT-L/14 CLIP teacher model.

1. Please unzip `PromptKD-main.zip` and go to this project.

2. Change `./scripts/promptsrc/base2new_train.sh line 11 CFG=vit_b16_c2_ep20_batch4_4+4ctx` to `vit_l14_c2_ep20_batch8_4+4ctx`.

3. Follow the instructions listed in `./docs/PromptSRC.md` and run the script.

### Running MTKD

1. The base-to-novel experimental settings are provided in the config file at `./configs/mtkd.yaml`.

2. If you want to run the first phase of MTKD, please delete the argument `--second-phase` in `./scripts/mtkd.sh`.
   If you want to run the second phase of MTKD, please add argument `--second-phase` in `./scripts/mtkd.sh` and type in the directory of the output results of 5-shot training in the first phase in line 144 of file `train.py`.

3. Run the commands below to train PromptKD on the specified dataset.

For example:
```
sh scripts/mtkd.sh dtd 1 1 8 0.005 0.25 24 0.2
```

4. The output results will be automatically saved at `output/${DATASET}/shots_${SHOTS}/MTKD/seed_${SEED}`.
