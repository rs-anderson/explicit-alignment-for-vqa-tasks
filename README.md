# Explicit alignment for few-shot Visual Question Answering

This is the research repository of the Vision Encoders in Visual Question Answering project, by Ryan Anderson.


### Table of Contents
<!-- TOC -->

- [Overview](#overview)
    - [Structure](#structure)
    - [Configs](#configs)
    - [ModuleParser](#moduleparser)
    - [MetricsProcessor](#metricsprocessor)
    - [NOTE](#note)
- [Environments](#environments)
- [Download Datasets](#download-datasets)
    - [COCO images](#coco-images)
    - [OKVQA Dataset](#okvqa-dataset)
- [Replicating results in report](#replicating-results-in-report)
    - [RA-VQA-NoDPR T5 baseline](#ra-vqa-nodpr-t5-baseline)
- [Additional experiments not included in report](#additional-experiments-not-included-in-report)
    - [RA-VQA-NoDPR T5 baseline](#ra-vqa-nodpr-t5-baseline)

<!-- /TOC -->


## Overview
The training and testing are backboned by pytorch-lightning. The pre-trained Transformer models are from Huggingface-transformers. The training platform is Pytorch.

### Structure
The framework consists of:

1. **main.py**: the main program. It loads a config file and override some entries with command-line arguments. It initialises a data loader wrapper, a model trainer, and a pytorch-lightning trainer to execute training and testing.
2. **Data Loader Wrapper**: it loads the data according to `data_modules` defined in config files. `.set_dataloader()` is called after data loading is finished. `.train_dataloader` and `.test_dataloader` are loaded.
3. **Datasets**: they are automatically loaded by the data loader wrapper. `.collate_fn` is defined to collate the data. An decorator class `ModuleParser` is used to help generate the training inputs. This decorator class generates input dict according to configs (`config.model_config.input_modules/decorder_input_modules/output_modules`).
4. **Model Trainers**: a pytorch-lightning `LightningModule` instance. It defines training/testing behaviors (training steps, optimizers, schedulers, logging, checkpointing, and so on). It initialises the model being trained at `self.model`.
5. **Models**: pytorch `nn.Modules` models.

### Configs
The configuration is achieved with `jsonnet`. It enables inheritance of config files. For example, `RAVQA.jsonnet` override its configs to `RAVQA_base.jsonnet`, which again inherits from `base_env.jsonnet` where most of important paths are defined.

By including the corresponding key:value pair in the config file, overriding can be easily performed.

### ModuleParser
A decorator class that helps to parse data into features that are used by models.

An example is shown below:
```
"input_modules": {
    "module_list":[
    {"type": "QuestionInput",  "option": "default", 
                "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},  
    {"type": "TextBasedVisionInput",  "option": "caption",
                "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
    {"type": "TextBasedVisionInput",  "option": "object", 
                "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
    ],
    "postprocess_module_list": [
    {"type": "PostProcessInputTokenization", "option": "default"},
    ],
},
"decoder_input_modules": {
    "module_list":[],
    "postprocess_module_list": [],
},
"output_modules": {
    "module_list":[
    {"type": "GenerationOutput", "option": "default"},
    ],
    "postprocess_module_list": [
    {"type": "PostProcessOutputTokenization", "option": "default"},
    ],
},
```
which first generates text_sequences:
```
<BOQ> Question <EOQ> <BOC> Caption <EOC> <BOV> obj1 attr1 attr2 <SOV> obj2 ... [OCR results] <EOV>
```
in the order defined in `input_modules`, and then the postprocessing unit `PostProcessInputTokenization` is used to tokenize the input into `input_ids` and `input_attention_masks`.

By defining new functions in `ModuleParser`, e.g. `self.TextBasedVisionInput`, a new behavior can be easily introduced to transform modules into training features.

### MetricsProcessor
The following entries in config file `test.metrics` define the metrics to compute in validation and testing. Each module uploads `log_dict` with `metrics_name: metrics_value` which can be processed in trainers conveniently.
```
"metrics": [
    {'name': 'compute_exact_match'},
    {'name': 'compute_retrieval_metrics'},
    {'name': 'compute_okvqa_scores'},
],
```

### NOTE
This framework is designed for **research purpose**, with flexibility for extension. It is not a perfect framework for production, of course.

## Environments
Create virtualenv:
```
python3 -m venv .venv
source .venv/bin/activate 
```
Note, we used `python3.8`.

Install Pytorch:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Install other libraries:
```
pip install -r requirements.txt
```


## Download Datasets
### COCO images
`data\ok-vqa\train2014`: [Train images](http://images.cocodataset.org/zips/train2014.zip)

`data\ok-vqa\val2014`: [Test images](http://images.cocodataset.org/zips/val2014.zip)

### OKVQA Dataset
`data\ok-vqa\mscoco_train2014_annotations.json`: [Training annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)

`data\ok-vqa\mscoco_val2014_annotations.json`: [Testing annotations](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip)

`data\ok-vqa\OpenEnded_mscoco_train2014_questions.json`: [Training questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)

`data\ok-vqa\OpenEnded_mscoco_val2014_questions.json`: [Testing questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip)

## Replicating results in report
### RA-VQA-NoDPR (T5 baseline)
```
python main.py ../configs/baseline_T5.jsonnet \
    --mode train \
    --experiment_name OKVQA_RA-VQA-NoDPR  \
    --accelerator auto --devices auto  \
    --opts train.epochs=10  \
            train.batch_size=1  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.scheduler=linear
```


