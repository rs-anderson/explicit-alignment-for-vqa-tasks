local base_env = import 'RAVQA_base.jsonnet';


local override = {
  "model_config": {
    "modules": [
      "read_static_retrieval_results",
    ],
    "loss_ratio":{
      "nll_loss": 1,
      "additional_loss": 0,
      "rag_loss": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQAWithKnowledge",
    "dataset_type": "OKVQADataset",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'max_target_length':10,
      'num_knowledge_passages': 5,
    },
    "dataset_modules": {
      "module_list": [
        "LoadVinVLFeatures",
        "LoadGoogleOCRFeatures",
        "LoadOscarCaptionFeatures",
        "LoadOKVQAData",
        "LoadGoogleSearchPassageData",
        "LoadPretrainedDPROutputForGoogleSearchPassage",
      ],
      "module_dict":{
        "LoadPretrainedDPROutputForGoogleSearchPassage": {
          "option": "none",
        },
      },
    },
  },
};

std.mergePatch(base_env, override)
