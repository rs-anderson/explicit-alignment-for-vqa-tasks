local question_test_file = '../data/ok-vqa/OpenEnded_mscoco_val2014_questions.json';
local question_train_file = '../data/ok-vqa/OpenEnded_mscoco_train2014_questions.json';
local annotation_test_file = '../data/ok-vqa/mscoco_val2014_annotations.json';
local annotation_train_file = '../data/ok-vqa/mscoco_train2014_annotations.json';
local VinVL_features_train = "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv";
local VinVL_features_test = "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_testset_full/inference/vinvl_vg_x152c4/predictions.tsv";
local img_data_path_train = "/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/train2014";
local img_data_path_test = "/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/val2014";
local ocr_features_train = "../data/ok-vqa/pre-extracted_features/OCR/train";
local ocr_features_test = "../data/ok-vqa/pre-extracted_features/OCR/valid";
local knowledge_features_train = "../Experiments/Knowledge_Retriever_BERT_dim_768_inbatch_negative/test/test_evaluation/train_predictions.json";
local knowledge_features_test = "../Experiments/Knowledge_Retriever_BERT_dim_768_inbatch_negative/test/test_evaluation/test_predictions.json";

local passage_file_train = "../data/ok-vqa/pre-extracted_features/passages/okvqa_train_corpus.csv";
local passage_file_test = "../data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv";
local knowledge_passage_annotation_train = "../data/ok-vqa/pre-extracted_features/passages/retriever_train.json";
local knowledge_passage_annotation_valid = "../data/ok-vqa/pre-extracted_features/passages/retriever_testdev.json";
local knowledge_passage_annotation_test = "../data/ok-vqa/pre-extracted_features/passages/retriever_test.json";


local dataset_name = "OK-VQA";

local default_cache_folder = '../data/ok-vqa/cache';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 100;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 5e-5;
local gradient_accumulation_steps = 4;
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
    "ModelClass": "T5ForConditionalGeneration",
    "TokenizerClass": "T5Tokenizer",
    "TokenizerModelVersion": "t5-large",
    "ConfigClass": "T5Config",
    "ModelVersion": "t5-large",
    "pretrained": 1,
    "modules": [
      "add_self_without_knowledge",
    ],
    "SPECIAL_TOKENS":{
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOA>", "<EOA>", "<BOK>", "<EOK>"],
    },
    "input_modules": [
      {"type": "QuestionInput",  "option": "default"},
      {"type": "VisionInput",  "option": "text_only", 
              "object_max": 40,
              "attribute_max": 3, "attribute_thres":0.05,
              "ocr": 0},
      {"type": "KnowledgeInput",  "option": "default"},
    ],
    "decoder_input_modules": [],
    "output_modules": [
      {"type": "GenerationOutput", "option": "default"},
    ],
  },
  "cache":{
    "default_folder": default_cache_folder,
    "regenerate":{
      "vinvl_feature_preprocessed": 0,
      "ocr_feature_preprocessed": 0,
      "train_data_preprocessed": 0,
      "test_data_preprocessed": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQAWithKnowledge",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_target_length':10,
      'num_knowledge_passages': 20,
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
    "ocr_features": {
        "train": ocr_features_train,
        "test": ocr_features_test,
    },
    "img_data_path":{
        "train": img_data_path_train,
        "test": img_data_path_test,
    },
    "passage_file": {
      "train": passage_file_train,
      "test": passage_file_test,
    },
    "knowledge_passage_annotation":{
        "train": knowledge_passage_annotation_train,
        "valid": knowledge_passage_annotation_valid,
        "test": knowledge_passage_annotation_test,
    },
    "knowledge_features": {
        "train": knowledge_features_train,
        "test": knowledge_features_test,
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "BaselineExecutorWithKnowledge",
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
    "break_interval": break_interval,
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