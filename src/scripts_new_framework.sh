
# RAVQA framework
## Static retrieval
python main.py ../configs/RAVQA_static_retrieval.jsonnet --mode train --experiment_name test_new_framework --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear model_config.loss_ratio.additional_loss=1 data_loader.additional.num_knowledge_passages=5

## Retrieval from Google Search Passages
### RA-VQA-FrDPR
python main.py ../configs/RAVQA.jsonnet --mode train --experiment_name OKVQA_RA-VQA-FrDPR_FullCorpus --accelerator auto --devices auto --modules freeze_question_encoder force_existence --opts train.epochs=10 train.batch_size=4 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=8 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear data_loader.additional.num_knowledge_passages=5

### RA-VQA-NoPR
python main.py ../configs/RAVQA.jsonnet --mode train --experiment_name RA-VQA-NoPR --accelerator auto --devices auto --modules force_existence --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear model_config.loss_ratio.additional_loss=1 model_config.RAVQA_loss_type=NoPR data_loader.additional.num_knowledge_passages=5

### RA-VQA
python main.py ../configs/RAVQA.jsonnet --mode train --experiment_name OKVQA_RA-VQA_FullCorpus --accelerator auto --devices auto --modules force_existence --opts train.epochs=10 train.batch_size=4 valid.step_size=32 valid.batch_size=4 train.additional.gradient_accumulation_steps=8 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear model_config.loss_ratio.additional_loss=1 model_config.RAVQA_loss_type=Approach6 data_loader.additional.num_knowledge_passages=5
python main.py ../configs/RAVQA.jsonnet --mode test --experiment_name RA-VQA --accelerator auto --devices auto --modules force_existence --opts data_loader.additional.num_knowledge_passages=1 data_loader.dummy_dataloader=1 test.load_model_path=../Experiments/RA-VQA/train/saved_model/epoch_01.ckpt

### RA-VQA-NoCT
python main.py ../configs/RAVQA.jsonnet --mode train --experiment_name RA-VQA-NoCT --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear model_config.loss_ratio.additional_loss=1 model_config.RAVQA_loss_type=Approach6 data_loader.additional.num_knowledge_passages=5

### RA-VQA on Wikipedia
python main.py ../configs/RAVQA_wikipedia.jsonnet --mode train --experiment_name RA-VQA_Wikipedia --accelerator auto --devices auto --modules force_existence --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear model_config.loss_ratio.additional_loss=1 model_config.RAVQA_loss_type=Approach6 data_loader.additional.num_knowledge_passages=5

# TRiG
python main.py ../configs/TRiG.jsonnet --mode train --experiment_name TRiG --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear data_loader.additional.num_knowledge_passages=2

# RA-VQA-NoDPR (T5 baseline)
python main.py ../configs/baseline_T5.jsonnet --mode train --experiment_name RA-VQA-NoDPR --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear

# T5 baseline with Knowledge
## Random pick
python main.py ../configs/baseline_T5_with_knowledge.jsonnet --mode train --experiment_name RA-VQA-NoDPR_with_random_pick_passage --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear data_loader.additional.num_knowledge_passages=5
## With all passages in training
python main.py ../configs/baseline_T5_with_knowledge.jsonnet --mode train --experiment_name RA-VQA-NoDPR_with_all_passages --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=1 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=32 train.lr=0.00006 train.retriever_lr=0.00001 train.scheduler=linear data_loader.additional.num_knowledge_passages=5 data_loader.dataset_modules.module_dict.LoadPretrainedDPROutputForGoogleSearchPassage.option=default data_loader.dataset_type=OKVQADatasetWithAllPassages


# DPR
python main.py ../configs/DPR.jsonnet --mode train --experiment_name OKVQA_DPR_FullCorpus --accelerator auto --devices auto --opts train.epochs=10 train.batch_size=30 valid.step_size=1 valid.batch_size=32 train.additional.gradient_accumulation_steps=2 train.lr=0.00001

python main.py ../configs/DPR.jsonnet --mode test --experiment_name OKVQA_DPR_FullCorpus --accelerator auto --devices auto --test_evaluation_name generate_test_set --opts train.batch_size=64 valid.batch_size=64 test.load_model_path=/home/wl356/rds/rds-cvnlp-hirYTW1FQIw/wl356/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/model_05.ckpt
python main.py ../configs/DPR.jsonnet --mode test --experiment_name OKVQA_DPR_FullCorpus --accelerator auto --devices auto --test_evaluation_name generate_train_set --opts train.batch_size=64 valid.batch_size=64 test.load_model_path=/home/wl356/rds/rds-cvnlp-hirYTW1FQIw/wl356/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/model_05.ckpt data_loader.use_dataset=train