# GPSocio: A Transformer-based General-purpose Social Network Representation System
## Requirements

 - Python 3.10.10
 - pandas 2.2.2
 - numpy 1.26
 - scikit-learn 1.4.2
 - PyTorch 2.0.0
 - PyTorch Lightning 2.0.0
 - Transformers 4.28.0
 - Deepspeed 0.9.0

## GPSocio Pret-train

1. Download backbone model `allenai/longformer-base-4096` and adjust Longformer for our experiment setting.
```bash
python save_longformer_ckpt.py
```
2. Pretrain GPSocio by running the following command. You can substute the data under `pretrain_data` folder following the same format to your own data.
```bash
bash lightning_run.sh
```
3. Go to the folder you save your model. If you use the training strategy `deepspeed_stage_2` (default setting in the script), you need to first convert zero checkpoint to lightning checkpoint by running `zero_to_fp32.py` (automatically generated to checkpoint folder from pytorch-lightning):
```bash
python zero_to_fp32.py . pytorch_model.bin
```
4. Convert the lightning checkpoint to pytorch checkpoint by running `convert_pretrain_ckpt.py`. You need to set four paths in the file: 
- `LIGHTNING_CKPT_PATH`, pretrained lightning checkpoint path.
- `LONGFORMER_CKPT_PATH`, Longformer checkpoint (from `save_longformer_ckpt.py`) path.
- `OUTPUT_CKPT_PATH`, output path of GPSocio checkpoint (for class `GPSocioModel` in `gpsocio/models.py`).
- `OUTPUT_CONFIG_PATH`, output path of GPSocio for Sequential Recommendation checkpoint (for class `GPSocioForSeqRec` in `gpsocio/models.py`). 

```bash
python convert_pretrain_ckpt.py
```

## Target Domain Representation Generation

1. In `test.sh` file, substitute the `pretrain_ckpt` `.bin` file you save your model you assigned in `OUTPUT_CONFIG_PATH` in `convert_pretrain_ckpt.py`, then run the following command:

```bash
bash test.sh
```

## Domain-specific Representation Adaptation

1. To update embedding for user prediction and sentiment analysis task, run the following command:
```bash
python task_utils/emb_propagate.py
```
You can adjust the `ratio` parameter to adjust the propagation weight.

2. To update embedding for assertion prediction task, run the following command:
```bash
python task_utils/emb_propagate_asser_pred.py
```
You can adjust the `ratio` parameter to adjust the propagation weight.

## Downstream Tasks Evaluation

1. To evaluate "user prediction" task, please run:
```bash
python task_evaluations/user_pred.py
```
2. To evaluate "assertion prediction" task, please run:
```bash
python task_evaluations/user_pred.py
```
3. To evaluate "sentiment analysis" task, please run:
```bash
python task_evaluations/sentiment_analysis.py
```

