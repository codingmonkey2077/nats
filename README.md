# Neural Attention Search


This repository contains the code for the submission "Neural Attention Search".

To train a new nats model, please run the following commands:
```
cd experiments
python train.py model.base_dir=\YOUR\PATH\TO\SAVE\MODEL n_gpus=2 dataset.base_dir=\YOUR\PATH\TO\DATASET transformer_args.nats_enable=True
```
Then you could evaluate the nats model with
```
cd experiments
python eval.py model.base_dir=\YOUR\PATH\TO\SAVE\MODEL n_gpus=2 dataset.base_dir=\YOUR\PATH\TO\DATASET transformer_args.nats_enable=True
```

To fine-tune a model, we need to first generate the fine-tuning training dataset.
Some of the datasets are from Huggingface, while the others need to be collected manually from the corresponding pages:
```
http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
https://github.com/StonyBrookNLP/musique
https://github.com/baidu/DuReader/tree/master/DuReader-2.0
https://gov-report-data.github.io/
https://github.com/Yale-LILY/QMSum
https://github.com/hahahawu/VCSum
https://github.com/Leolty/repobench
```
And the synthetic dataset
```
https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/fine-tune/booksum.jsonl.zst
```
Once all the dataset is downloaded, please run: 
```
cd experiments/finetune_datasets
python prepare_longbench_train_data.py  --long_bench_dataset_path \PATH\TO\THE\DOWNLOADED\DATASET \
                                        --dataset YOURDATASET \ 
                                        --res_dir \PATH\THAT\YOU\WOULD\LIKE\TO\STORE\THE\DATA \
                                        --tokenizer_path \LLM\PATH
```
and then download the synthetic dataset towards 
```
cd \PATH\THAT\YOU\WOULD\LIKE\TO\STORE\THE\DATA
wget https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/fine-tune/booksum.jsonl.zst
```
Now you could fine-tune a model on the generated dataset (we currently support Llama and Mistral model families)
by customizing the corresponding configurations under `experiments/configs/finetune_distill`
```
cd experiments
python hf_finetune_longbench.py 
```

Alternatively, we provide one adapter under the adapters folder; you could directly run the experiments with the provided adapter:
```
cd experiments/long_bench
pyhton hf_pred.py --nats_enable --adapter_path adapters --model_name Meta-Llama-3.1-8B-Instruct --adapter_name nats_3e7_SWindowlen256_lr0.002_wd0.1_train7_mixed
```
