local base_env = import 'base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local retriever_lr = 1e-5;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;



local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "T0_3B",
    "ModelClass": "VCT0Prefix",
    "TokenizerClass": "AutoTokenizer",
    "TokenizerModelVersion": "bigscience/T0_3B",
    "ConfigClass": "T0_3B",
    "ModelVersion": "",
    "pretrained": 1,
    "modules": [
    ],
    "model_args": {
        prefix_length: 10,
        // clip_length: 10,
        prefix_size: 768,  # dimensions of clip embedding
        mapping_type: "mlp",  # "perceiver" or "transformer" or "mlp"
        num_layers: 8,
        model_version: "bigscience/T0_3B",
    },
    "SPECIAL_TOKENS":{
    //   "bos_token": "<BOS>",
    //   "pad_token": "<PAD>",
      "additional_special_tokens": [],
    },
    "input_modules": {
      "module_list":[
        // {"type": "QAInput",  "option": "default", 
        //           "separation_tokens": {'start': '', 'end': ''}},
        {"type": "EmbeddingInput",  "option": "default"},          
      ],
      "postprocess_module_list": [
        {"type": "PostProcessInputTokenization", "option": "default"},
        {"type": "PostProcessClipEmbeddings", "option": "default"},
      ],
    },
    "decoder_input_modules": {
      "module_list":[
        {"type": "QInput",  "option": "default", 
                  "separation_tokens": {'start': '', 'end': ''}},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessInputTokenization", "option": "generation"},
      ],
    },
    "output_modules": {
      "module_list":[
        {"type": "GenerationOutput", "option": "default"},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessOutputTokenization", "option": "default"},
      ],
    },
  },
  "cache":{
    "regenerate":{
      "train_data_preprocessed": 0,
      "val_data_preprocessed": 0,
      "clip_embeddings": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderConceptualCaptions",
    "dataset_type": "",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':1024,
      'max_decoder_source_length': 1024,
      'max_target_length': 100,
    },
    "dataset_modules": {
      "module_list": [
        "LoadConceptualCaptions",
      ],
      "module_dict":{
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "VCT0Executor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "retriever_lr": retriever_lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "save_interval":save_interval,
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "break_interval": break_interval,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "additional": {
        "multiprocessing": 4,
    },
  },
  "metrics": [
    {'name': 'do_nothing_metric'},
  ],
};

std.mergePatch(base_env, override)
