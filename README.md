# Explicit alignment for few-shot Visual Question Answering

This is the research repository for the project titled **_Vision Encoders in Visual Question Answering_**, by Ryan Anderson.


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
    - [Conceptual Captions](#conceptual-captions)
    - [VQA2](#vqa2)
- [In-context example selection](#in-context-example-selection)
- [Replicating results in report](#replicating-results-in-report)
    - [Training mapping network on Conceptual Captions](#training-mapping-network-on-conceptual-captions)
    - [Evaluating few-shot performance on VQA2](#evaluating-few-shot-performance-on-vqa2)
- [Additional arguments for main](#additional-arguments-for-main)

<!-- /TOC -->


## Overview
The training and testing are backboned by pytorch-lightning. The pre-trained Transformer models are from Huggingface-transformers. The training platform is Pytorch.

![high-level-overview](https://github.com/rs-anderson/explicit-alignment-for-vqa/blob/main/figures/high-level-overview.png)


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
    {'name': 'compute_vqa_scores'},
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

### Conceptual captions

We downloaded the conceptual captions dataset from HuggingFace. Because this is a large dataset, we extracted the CLIP vision encodings for each item when downloading the dataset. Run the file `/src/tools/extract_clip_embeddings_conceptual_captions.py`. The resulting CLIP embeddings should be stored in `/data/conceptual_captions/pre-extracted-features`. Run `/data/conceptual_captions/pre-extracted-features/convert_str_columns_to_list.py` to apply the final formatting to the captions.

### VQA2

All data for the VQA2.0 task can be downloaded [here](https://visualqa.org/download.html). The repo expects the data to be distributed into the following directory structure.
#### COCO images

`data\vqa2\train2014`: [Train images](http://images.cocodataset.org/zips/train2014.zip)

`data\vqa2\val2014`: [Test images](http://images.cocodataset.org/zips/val2014.zip)

#### VQA2 dataset
`data\vqa2\v2_mscoco_train2014_annotations.json`: [Training annotations](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip)

`data\vqa2\v2_mscoco_val2014_annotations.json`: [Testing annotations](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)

`data\vqa2\v2_OpenEnded_mscoco_train2014_questions.json`: [Training questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)

`data\vqa2\v2_OpenEnded_mscoco_val2014_questions.json`: [Testing questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)


#### CLIP embeddings

Use `/src/tools/extract_contrastive_image_embeddings.py` to extract the CLIP vision encodings for the VQA2.0 training and validation images. This should be written to `/data/vqa2/pre-extracted_features/clip_embeddings`.

Use `/src/tools/extract_contrastive_text_embeddings.py` to extract the CLIP text encodings for the VQA2.0 training and validation questions. This should be written to `/data/vqa2/pre-extracted_features/text_embeddings`


## In-context example selection

**RICES**

You will have to setup the FAISS index with conda. See [this](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss). We extracted the in-context examples in four steps:

1. Run `src/in_context_example_selection/get_question_knns.py`.
2. Run `src/in_context_example_selection/reformat_faiss_output.py`
3. Run `src/in_context_example_selection/get_image_knn_from_text_knn.py`
4. Run `src/in_context_example_selection/get_average_similarities.py`

Please note that you will need to change the file paths used in each of these scripts.

**Text-only RICES**

When running step 4 above, set `rices_for_image_and_question = False` and `rices_for_question_only = True` in `src/in_context_example_selection/get_average_similarities.py`

**RANDOM**

Use the script `src/utils/in_context_examples.py`.


The in-context examples selected for a specific VQA question_id can be visualised (like in Figure 5.1) using the script `src/tools/visualise_in_context_examples.py` using the following command:
```
python visualise_in_context_examples.py --question_id 262677001 --in_context_examples_fname rices.pkl --num_in_context_examples 4
```

## Replicating results in report


The following scripts can be used to replicate the results in the report. Please ensure that you have all of the necessary clip embeddings and in-context examples stored at `data/vqa2/pre-extracted_features`. For example, my directory looks like this
```
pre-extracted_features/
  clip_embeddings/
     coco_ViT-L_14@336px_train2014.pkl
     coco_ViT-L_14@336px_val2014.pkl
  in_context_examples/
     rices.pkl
     rices_questions_only.pkl
     random.pkl
  text_embeddings/
     coco_ViT-L_14@336px_train2014.pkl
     coco_ViT-L_14@336px_val2014.pkl
```

### Training mapping network on Conceptual Captions

In order to train the mapping network of T0-3B (n=10), run the following command:

```
python main.py \
    ../configs/conceptual_captions/conceptual_captions.jsonnet \
    --mode train \
    --experiment_name VC-T0-Conceptual-Captions-Test \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=64  valid.step_size=1  valid.batch_size=64  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

Change the `prefix_length: 10` to `prefix_length: 5` if you want to train T0-3B (n=5).

The trained model can then be used to generate captions using `src/generate_captions.ipynb`.

### Evaluating few-shot performance on VQA2

**Replicating best results (Table 6.1)**

Row 1:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots 1 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

Row 2:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_frozen.jsonnet \
    --num_shots 1 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

**Replicating zero-shot performance (Figure 6.2)**

zero-shot hotpotqa:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots 0 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

zero-shot frozen:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_frozen.jsonnet \
    --num_shots 0 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

**Replicating few-shot performance (Figure 6.5)**

Repeat the following scripts for k = 0, 1, 2, 4, 8.

k-shot hotpotqa:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

k-shot frozen:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_frozen.jsonnet \
    --num_shots k \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

**Replicating importance of in-context example selection (Figure 6.6)**

Repeat the following scripts for k = 0, 1, 2, 4, 8.

RICES:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

Random:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/random.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

**Replicating importance of visual prefix (Figure 6.7)**

Repeat the following scripts for k = 0, 1, 2, 4.

Default:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

Text-only prompt:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --no_prefix 1 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

Text-only prompt with text-only RICES:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --no_prefix 1 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices_questions_only.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

**Replicating prompt ensembling results (Figure 6.8)**

Repeat the following scripts for k = 2, 4.

No ensemble:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

Ensemble:
```
python main.py \
    ../configs/vqa2/few_shot_vqa_hotpotqa.jsonnet \
    --num_shots k \
    --num_permutations_of_in_context_examples 5 \
    --in_context_examples_fpath ../data/vqa2/pre-extracted_features/in_context_examples/rices.pkl \
    --mode test \
    --experiment_name EXPERIMENT_NAME \
    --accelerator auto \
    --devices auto \
    --log_prediction_tables \
    --opts train.epochs=10  train.batch_size=16  valid.step_size=1  valid.batch_size=128  train.additional.gradient_accumulation_steps=2  train.lr=0.0001 check_val_every_n_epoch=1
```

## Additional arguments for main

There are a number of command-line arguments that can be passed to `main.py` in order to implement other experiments.

`--no_prefix 1` - the visual prefix is removed. 

`--pass_examples_through_encoder_one_at_a_time 1` - each in-context example is encoded individually, then the encodings are concatenated and passed to the decoder. 

`--num_permutations_of_in_context_examples 5 ` - permute the in-context examples 5 times and then ensemble results.
    
`--sample_templates 1 ` - sample qa templates when formatting inputs.

`--ensemble_one_shots 1 `- convert the K-shot evaluation into K 1-shot predictions and then ensemble the K predictions.

We did not find the results for these experiments to be sufficiently noteworthy.
