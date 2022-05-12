# Add parent directory to python path to access lightning_base.py
# export PYTHONPATH="../":"${PYTHONPATH}"

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options



python finetune_rag.py \
    --data_dir /home/wl356/projects/Knowledge-based-visual-question-answering/src \
    --output_dir ./output_pretrained_DPR_sequence \
    --model_name_or_path ./pretrained_dpr_initial_T5_large \
    --model_type rag_sequence \
    # --passages_path ../rag-end2end-retriever/ok-vqa-passages-no-split-pretrained/my_knowledge_dataset \
    # --index_path  ../rag-end2end-retriever/ok-vqa-passages-no-split-pretrained/my_knowledge_dataset_hnsw_index.faiss \
    # --index_name custom \
    --gpus 1 \
    --profile \
    --do_train \
    --do_predict \
    --n_train -1 \
    --n_val -1 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_source_length 512 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.0 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler linear \
    --learning_rate 1e-04 \
    --num_train_epochs 15 \
    --warmup_steps 0 \
    --gradient_accumulation_steps 8 \
