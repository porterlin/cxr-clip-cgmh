# CXR-CLIP-CGMH
## Prepare dataset
訓練和驗證資料集的 csv 檔要包含以下欄位
| index | image              | text_label        |
|-------|--------------------|-------------------|
| 0     | List of image_path | [positive labels] |

測試資料集的 csv 檔要包含以下欄位
| index | image              | label_names        | text        |
|-------|--------------------|-------------------|-------------------|
| 0     | List of image_path | 0 or 1 for each label_name | text |
## Setup
### Dataset
將以下檔案中的 data_path 替換成自己電腦中長庚資料集的路徑
```
configs/data_train/cgmh.yaml
configs/data_valid/cgmh.yaml
configs/data_test/cgmh.yaml
```

### Prompts
```datasets/train_prompts_cgmh.json``` 定義如何把 label 轉換成句子。範例如下，每個 label 會有 3 個部分分別為 pos, neg, unc (positive, negative, uncertain)。自行填入句子，訓練時會從中隨機挑選。
```json
"Pneumothorax": {
      "pos": [
        "pneumothorax.",
        "there is pneumothorax.",
        "pneumothorax is present.",
        "pneumothorax is seen.",
        "pneumothorax is noted.",
        "the presence of pneumothorax is seen.",
        "the presence of pneumothorax is noted.",
        "Tension pneumothorax."
      ],
      "neg": [
        "no pneumothorax.",
        "no evidence of pneumothorax.",
        "no convincing evidence of pneumothorax.",
        "no definite evidence of pneumothorax.",
        "no visible pneumothorax."
      ],
      "unc": [
        "Pneumothorax could be present.",
        "Pneumothorax might be present.",
        "Pneumothorax may be present.",
        "Pneumothorax possibly be present."
      ]
    }
```
目前預設是只取pos。比方說如果有個病人同時有 Pneumothorax 和 Pigtail malpositon，訓練時會從 Pneumothorax 和 Pigtail malpositon 的 pos 中，各自隨機選擇一個句子。

如果需要 neg 可以修改 ```configs/data_train/cgmh.yaml```, ```configs/data_valid/cgmh.yaml``` 中的 num_negs 值。

假設 num_negs=3，則根據上述例子會從 Pneumothorax 和 Pigtail malpositon 之外的其他疾病隨機選擇 3 個，然後在該疾病下的 neg 隨機選擇一個。

最後會把所有選到的句子合起來當作該病人的報告

### Optional
如果需要更換 label，```cxrclip/prompt/constants.py``` 中的 CGMH_TASKS 也要做修改

## Train
* 單機多 GPU
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=NVL torchrun --standalone --nproc_per_node=4 train.py
  ```
  * CUDA_VISIBLE_DEVICES: 程式可以看到的 GPU
  * NCCL_P2P_LEVEL: 定義 GPU 間的通訊機制
  * --standalone: 單機
  * --nproc_per_node: process 數量，原則上一顆 GPU 一個 process。所以 CUDA_VISIBLE_DEVICES 有幾顆這裡就設幾個
* sigle gpu
  ```bash
  python train.py
  ```

## Eval
* 測試整個資料集
  ```bash
  CUDA_VISIBLE_DEVICES=0 python evaluate_clip.py test.checkpoint="/PATH/model-best.tar"
  ```
* 單張圖片推理
  ```bash
  CUDA_VISIBLE_DEVICES=0 python inference.py --model_path /PATH/model-best.tar --image_path /PATH/image.jpg
  ```

***
# CXR-CLIP 原始 README
This is an official Pytorch Implementation of **"CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training"** [[arxiv]](https://arxiv.org/abs/2310.13292)

## Environment setup
We have experimented the implementation on the following enviornment.
- Pytorch 1.12
- CUDA 11
```bash
pip install -r requirements.txt
```

## Prepare dataset
Datasets we used are as follows:

|           Dataset |                                                                            Download |              Comment |
|:-----------------:|:-----------------------------------------------------------------------------------:|----------------------|
| MIMIC-CXR         | [Link](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)                          | official split       |
| CheXpert          | [Link](https://stanfordmlgroup.github.io/competitions/chexpert/)                    | official split for train and val, and `chexpert_5x200` from [GLoRIA](https://stanfordmedicine.app.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh) for test |
| ChestX-ray14      | [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC)                                 | not used for test    |
| VinDr-CXR         | [Link](https://physionet.org/content/vindr-cxr/1.0.0/)                              | official split for test, and random split for train and val |
| RSNA-Pneumonia    | [Link](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) | same split as [GLoRIA](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L49-L50) |
| SIIM-Pneumothorax | [Link](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data) | same split as [GLoRIA](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L90-L91) |
| OpenI | [Link](https://openi.nlm.nih.gov/faq#collection) | all frontal images are used for evaluation |

For more details, please refer to [data preparation](datasets/README.md).

## Pre-trained model checkpoint
We trained Resnet50 and SwinTiny models with three dataset compositions.  
MIMIC-CXR (**M**), MIMIC-CXR + CheXpert (**M,C**), MIMIC-CXR + CheXpert + ChestX-ray14 (**M,C,C14**)

| model / dataset |  M  | M,C | M,C,C14 | 
|---------------|--------------------|------------------------|-|
| ResNet50      |  [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f982386ef38aa3ecd6ce1f8f928e8aa8/r50_m.tar)   |   [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f7ebbe4ad815868905d0820dbbde3662/r50_mc.tar)  | [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/de4b5e4ae2047c1fb7960ddcd8d861cb/r50_mcc.tar) |
| SwinTiny      |  [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a21ef120894e072ae77434daf6b98b72/swint_m.tar)   |   [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/97cbbdfb347d22ea44e95a70c7b0520a/swint_mc.tar)   | [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a25ce760064591c899f67c97ed7790de/swint_mcc.tar) |

## Pre-Train model
### command line
* single gpu
    ```bash
    python train.py {--config-name default_config}
    ```
* multi gpu
    ```bash
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=45678 train.py {--config-name default_config}
    ```

## Evaluation
### Zero-shot Evaluation
* Zero-shot classification
  * perform zero-shot and image-text retrieval evaluation
    on ( vindr_cxr, rsna_pneumonia, siim_pneumothorax, chexpert5x200, mimic_cxr, openi )
  ```bash
  python evaluate_clip.py test.checkpoint=${CKPT_PATH/model-best.tar}
  ```

### Fine-tuned Classifier (linear probing)
* on rsna_pneumonia
```bash
# train
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=rsna_pneumonia data_valid=rsna_pneumonia model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=rsna_pneumonia data_valid=rsna_pneumonia model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%
# evaluate
python evaluate_finetune.py data_test=rsna_pneumonia test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}
```
* on siim_pneumothorax
```bash
# train
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=siim_pneumothorax data_valid=siim_pneumothorax model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=siim_pneumothorax data_valid=siim_pneumothorax model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%
# evaluate
python evaluate_finetune.py data_test=siim_pneumothorax test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}
```
* on vindr_cxr
```bash
# train
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=vindr_cxr data_valid=vindr_cxr model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=vindr_cxr data_valid=vindr_cxr model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%
# evaluate
python evaluate_finetune.py data_test=vindr_cxr test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}
```
## Citation
```
@incollection{You_2023,
	doi = {10.1007/978-3-031-43895-0_10},
	url = {https://doi.org/10.1007%2F978-3-031-43895-0_10},
	year = 2023,
	publisher = {Springer Nature Switzerland},
	pages = {101--111},
	author = {Kihyun You and Jawook Gu and Jiyeon Ham and Beomhee Park and Jiho Kim and Eun K. Hong and Woonhyuk Baek and Byungseok Roh},
	title="CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training",
	booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
}
```
## License
CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training © 2023 is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1)

## Contact for Issues
Kihyun You, [kihyun.you@soombit.ai](kihyun.you@soombit.ai)  
Jawook Gu, [jawook.gu@soombit.ai](jawook.gu@soombit.ai)
