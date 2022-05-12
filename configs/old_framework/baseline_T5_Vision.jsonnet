local question_test_file = '../data/ok-vqa/OpenEnded_mscoco_val2014_questions.json';
local question_train_file = '../data/ok-vqa/OpenEnded_mscoco_train2014_questions.json';
local annotation_test_file = '../data/ok-vqa/mscoco_val2014_annotations.json';
local annotation_train_file = '../data/ok-vqa/mscoco_train2014_annotations.json';
local VinVL_features_train = "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_large_okvqa_trainset/inference/vinvl_large/predictions.tsv";
local VinVL_features_test = "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_large_okvqa_testset/inference/vinvl_large/predictions.tsv";
local img_data_path_train = "/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/train2014";
local img_data_path_test = "/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/val2014";

local dataset_name = "OK-VQA";
local default_cache_folder = '../data/ok-vqa/cache';

local train_batch_size = 64;
local valid_batch_size = 64;
local test_batch_size = 64;
local valid_step_size = 1;
local save_interval = 100;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 1;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;

{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "T5",
    "ModelClass": "T5ForConditionalGenerationWithVision",
    "FeatureExtractorClass": "ViTFeatureExtractor",
    "FeatureExtractorModelVersion": "google/vit-base-patch16-224-in21k",
    "TokenizerClass": "T5Tokenizer",
    "TokenizerModelVersion": "t5-large",
    "ConfigClass": "T5Config",
    "ModelVersion": "t5-large",
    "SPECIAL_TOKENS":{
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": ["<BOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOA>", "<EOA>"],
    },
    "pretrained": 1,
    "modules": [
    ],
    "input_modules": [
      {"type": "VisionInput",  "option": "region_of_interest"},
      // {"type": "VisionInput",  "option": "text_only"},
      {"type": "QuestionInput",  "option": "default"},
    ],
    "decoder_input_modules": [
    ],
    "output_modules": [
      {"type": "GenerationOutput", "option": "default"},
    ],
  },
  "cache":{
    "default_folder": default_cache_folder,
    "regenerate":{
      "vinvl_feature_preprocessed": 0,
      "train_data_preprocessed": 0,
      "test_data_preprocessed": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQA",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_target_length':50,
    },
    "annotation_files": {
        "train": annotation_train_file,
        "test": annotation_test_file,
    },
    "question_files": {
        "train": question_train_file,
        "test": question_test_file,
    },
    "VinVL_features": {
        "train": VinVL_features_train,
        "test": VinVL_features_test,
    },
    "img_data_path":{
        "train": img_data_path_train,
        "test": img_data_path_test,
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "DualEncoderExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch":-1,
    "save_interval":save_interval,
    "load_model_path": "",
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
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "load_model_path": "",
    "additional": {
        "multiprocessing": 4,
    },
  }
}