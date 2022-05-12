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
local caption_features = [
  "../data/ok-vqa/pre-extracted_features/captions/train_predictions.json",
  "../data/ok-vqa/pre-extracted_features/captions/valid_predictions.json",
  "../data/ok-vqa/pre-extracted_features/captions/test_predictions.json",
];
local knowledge_features_train = "/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/pre-extracted_features/knowledge/BERT_baseline/train_predictions.json";
local knowledge_features_test = "/home/wl356/projects/Knowledge-based-visual-question-answering/data/ok-vqa/pre-extracted_features/knowledge/BERT_baseline/test_predictions.json";

local passage_file_train = "../data/ok-vqa/pre-extracted_features/passages/okvqa_train_corpus.csv";
local passage_file_test = "../data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv";
local knowledge_passage_annotation_train = "../data/ok-vqa/pre-extracted_features/passages/retriever_train.json";
local knowledge_passage_annotation_valid = "../data/ok-vqa/pre-extracted_features/passages/retriever_testdev.json";
local knowledge_passage_annotation_test = "../data/ok-vqa/pre-extracted_features/passages/retriever_test.json";

local index_passages_path = "/home/wl356/rds/rds-wjb31-nmt2020/wl356/transformers/examples/research_projects/rag-end2end-retriever/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset";
local index_path = "/home/wl356/rds/rds-wjb31-nmt2020/wl356/transformers/examples/research_projects/rag-end2end-retriever/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset_hnsw_index.faiss";


local dataset_name = "OK-VQA";

local default_cache_folder = '../data/ok-vqa/cache';

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

{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "/home/wl356/rds/rds-wjb31-nmt2020/wl356/Experiments",
  "TENSORBOARD_FOLDER": "",
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "RAG",
    "ModelClass": "RagModel", // general class
    "TokenizerClass": "DPRQuestionEncoderTokenizer",  // question encoder tokenizer
    "TokenizerModelVersion": "facebook/dpr-question_encoder-single-nq-base", // question encoder tokenizer version
    
    "DecoderTokenizerClass": "T5Tokenizer",  // generator tokenizer
    "DecoderTokenizerModelVersion": "t5-large", // generator tokenizer version

    "QueryEncoderModelClass": "DPRQuestionEncoder", // question encoder
    "QueryEncoderConfigClass": "DPRConfig", // question encoder
    // "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "QueryEncoderModelVersion": "/home/wl356/rds/rds-wjb31-nmt2020/wl356/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/train/saved_model/epoch6/query_encoder",
    
    "GeneratorModelClass": "T5ForConditionalGeneration", // answer generator
    "GeneratorConfigClass": "T5Config",
    "GeneratorModelVersion": "t5-large",
    "pretrained": 1,
    "modules": [
      "rag_data_loading",
      // "read_static_retrieval_results",
      // "freeze_question_encoder",
      // "majority_voting",
      // "CE_loss_only",
    ],
    "loss_ratio":{
      "nll_loss": 1,
      "additional_loss": 0,
      "retrieval_pseudo_loss": 0,
      "rag_loss": 0,
    },
    "SPECIAL_TOKENS":{  // for query encoder
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
    "DECODER_SPECIAL_TOKENS":{ // for answer generator
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"],
    },
    "input_modules": [
      {"type": "QuestionInput",  "option": "default"},
      {"type": "VisionInput",  "option": "caption"},
      {"type": "VisionInput",  "option": "text_only", 
              "object_max": 40,
              "attribute_max": 3, "attribute_thres":0.05,
              "ocr": 1},
    ],
    "input_separation_tokens": {
      "question_input": {'start': '<BOQ>', 'end': '<EOQ>'},
      "caption_input": {'start': '<BOC>', 'end': '<EOC>'},
      'object_input': {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'},
      'knowledge_input': {'start': '<BOK>', 'end': '<EOK>'},
    },
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
      'max_decoder_source_length': 512,
      'max_target_length':10,
      'num_knowledge_passages': 5,
    },
    "index_files": {
      "index_passages_path": index_passages_path,
      "index_path": index_path,
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
    "caption_features": caption_features,
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
    "type": "RagExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "retriever_lr": retriever_lr,
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