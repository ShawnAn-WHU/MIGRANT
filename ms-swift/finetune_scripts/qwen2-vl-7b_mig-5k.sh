CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --train_type lora \
    --dataset /home/anxiao/Datasets/MIGRANT/mig_5k.json \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --save_strategy epoch \
    --save_steps 1 \
    --eval_steps 1 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir ../output_mig_5k \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4