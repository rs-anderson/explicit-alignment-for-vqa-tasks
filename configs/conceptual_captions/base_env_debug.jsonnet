// This is the base environment file
// It serves as default values for all other jsonnet config files
// Please override these values dirrectly in corresponding config files


// Default values for training control
local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;

// data path configuration
local wandb_cache_dir = '';
local default_cache_folder = '../data/conceptual_captions/cache';
local vqa_data = {
  "question_files":{
    "train": '../data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json',
    "val": '../data/vqa2/v2_OpenEnded_mscoco_val2014_questions.json',
  },
  "annotation_files": {
    "train": '../data/vqa2/v2_mscoco_train2014_annotations.json',
    "val": '../data/vqa2/v2_mscoco_val2014_annotations.json',
  },
};
local conceptual_captions = {
  "train": "../data/conceptual_captions/pre-extracted-features/conceptual_captions_ViT-L_14@336px_validation.parquet",
  "val": "../data/conceptual_captions/pre-extracted-features/conceptual_captions_ViT-L_14@336px_validation.parquet",
};
local img_data = {
  "train": "../data/ok-vqa/train2014",
  "val": "../data/ok-vqa/val2014",
};
local clip_embeddings = {
  "train": "../data/vqa2/pre-extracted_features/clip_embeddings/coco_clip-vit-base-patch32_train2014.pkl",
  "val": "../data/vqa2/pre-extracted_features/clip_embeddings/coco_clip-vit-base-patch32_val2014.pkl",
  "test": "../data/vqa2/pre-extracted_features/clip_embeddings/coco_clip-vit-base-patch32_test2015.pkl",
};
local VinVL_features = {
  "train": "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv",
  "test": "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_testset_full/inference/vinvl_vg_x152c4/predictions.tsv",
};
local ocr_features = {
  "train": "../data/ok-vqa/pre-extracted_features/OCR/train",
  "test": "../data/ok-vqa/pre-extracted_features/OCR/valid",
  "combine_with_vinvl": true,
};
local caption_features = {
  "train": "../data/ok-vqa/pre-extracted_features/captions/train_predictions.json",
  "valid": "../data/ok-vqa/pre-extracted_features/captions/valid_predictions.json",
  "test": "../data/ok-vqa/pre-extracted_features/captions/test_predictions.json",
};
local passage_data = {
  "train": "../data/ok-vqa/pre-extracted_features/passages/okvqa_train_corpus.csv",
  "full": "../data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv",
};
local pretrained_dpr_features = {
  "train": "../Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/test/test_evaluation/train_predictions.json",
  "test": "../Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/test/test_evaluation/test_predictions.json",
};
local dpr_training_annotations = {
  "train": "../data/ok-vqa/pre-extracted_features/passages/retriever_train.json",
  "valid": "../data/ok-vqa/pre-extracted_features/passages/retriever_testdev.json",
  "test": "../data/ok-vqa/pre-extracted_features/passages/retriever_test.json",
};

{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "WANDB": {
    "CACHE_DIR":  wandb_cache_dir,
    "entity": "byrne-lab",
    "project": "VQA (Ryan)",
    "tags": ["VQA2-clipcap"],
  },
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "RAG",
    "pretrained": 1,
    "modules": [],
    "input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "rag_modules": {
      "module_list":[],
    },
    "decoder_input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "output_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
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
    "dataset_type": "OKVQADataset",
    "dummy_dataloader": 0,
    "additional":{},
    "dataset_modules": {
      "module_list": [],
      "module_dict":{   // all available modules
        "LoadVinVLFeatures":{
          "type": "LoadVinVLFeatures", "option": "default", 
          "config": VinVL_features,
        },
        "LoadGoogleOCRFeatures":{
          "type": "LoadGoogleOCRFeatures", "option": "default",
          "config": ocr_features,
        },
        "LoadClipEmbeddings":{
          "type": "LoadClipEmbeddings", "option": "default",
          "config": clip_embeddings,
        },
        "LoadOscarCaptionFeatures": {
          "type": "LoadOscarCaptionFeatures", "option": "default",
          "config": caption_features,
        },
        "LoadConceptualCaptions": {
          "type": "LoadConceptualCaptions", "option": "default",
          "config": {
            "conceptual_captions_path": conceptual_captions,
          },
        },
        "LoadVQA2Data": {
          "type": "LoadVQA2Data", "option": "default",
          "config": {
            "vqa_data_path": vqa_data,
            "image_data_path": img_data,
          },
        },
        "LoadGoogleSearchPassageData": {
          "type": "LoadGoogleSearchPassageData", "option": "default",
          "config": {
            "passage_data_path": passage_data,
            "use_full_split": true,
          },
        },
        "LoadPretrainedDPROutputForGoogleSearchPassage": {
          "type": "LoadPretrainedDPROutputForGoogleSearchPassage", "option": "none",
          "config": {
            "pretrained_dpr_outputs": pretrained_dpr_features,
          },
        },
        "LoadGoogleSearchAnnotations": {
          "type": "LoadGoogleSearchAnnotations", "option": "default",
          "config": {
            "annotations_path": dpr_training_annotations,
          },
        },
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "RagExecutor",
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
        "plugins": [],
        "save_top_k": 1,
        "save_top_k_metric": "test/loss",
        "save_top_k_mode": "min",
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