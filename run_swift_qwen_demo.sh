# ***** 对Qwen2.5-7B-Instruct进行自我认知微调 (22GB 可以直接运行) ***** #
# demo来自ms-swift官网快速开始部分 ：https://github.com/modelscope/ms-swift/blob/main/README_CN.md

# 通过参数CUDA_VISIBLE_DEVICES 设置分布式 or 单机，比如
  # 分布式: CUDA_VISIBLE_DEVICES=0,1,2,3 
  # 单机: CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
