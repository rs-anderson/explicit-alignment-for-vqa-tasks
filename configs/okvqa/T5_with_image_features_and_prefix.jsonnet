local base_env = import 'T5_with_image_features.jsonnet';

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
  "model_config": {
    "base_model": "T5",
    "ModelClass": "T5ForConditionalGenerationWithVision",
    "TokenizerClass": "T5Tokenizer",
    "TokenizerModelVersion": "t5-large",
    "ConfigClass": "T5Config",
    "ModelVersion": "t5-large",
    "pretrained": 1,
    "ImagePreprocessorClass": "EfficientNetImagePreprocessor",
    "ImagePreprocessorConfig": {
        'input_size': [3, 224, 224],
        'interpolation': 'bicubic', 
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225], 
        'crop_pct': 0.875
    },
    "modules": [
    ],
    "SPECIAL_TOKENS":{
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>"],
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {'start': 'question:', 'end': 'context:'}},
        {"type": "TextBasedVisionInput",  "option": "caption",
                  "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
        {"type": "TextBasedVisionInput",  "option": "object", 
                  "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                  "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
        {"type": "ImageInput",  "option": "default"},          
      ],
      "postprocess_module_list": [
        {"type": "PostProcessInputTokenization", "option": "default", "task_prefix": ""},
        {"type": "PreProcessImage", "option": "default"},
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
  },
};

std.mergePatch(base_env, override)
