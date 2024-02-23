model=meta-llama/Llama-2-7b-hf
lr=2e-5
batch_size=128
lora_rank=512
split_batch_size=2
gpu_num=4
epoch=3
data_path=./data/Llama-2-7b-hf/abs_rdm_data_ins200low.jsonl
dir_path=[DIR_PATH] # your output dir

accumulation_steps=`expr ${batch_size} / ${split_batch_size} / ${gpu_num}`
torchrun --nproc-per-node ${gpu_num} generator_main.py \
--model_name_or_path ${model} --output_dir ${dir_path} \
--do_train --do_eval --per_device_train_batch_size ${split_batch_size} \
--learning_rate ${lr} --gradient_accumulation_steps ${accumulation_steps} \
--num_train_epochs ${epoch} --report_to none --load_from_pretrain True \
--train_data_path ${data_path} --valid_data_path ${data_path} \
--max_valid_samples 150 --max_length 512 \
--evaluation_strategy no --save_strategy no \
--per_device_eval_batch_size 1 --eval_accumulation_steps 1 \
--lora_rank ${lora_rank} --lr_scheduler_type cosine \
--predict_with_generate --warmup_ratio 0.03 --torch_dtype bfloat16
