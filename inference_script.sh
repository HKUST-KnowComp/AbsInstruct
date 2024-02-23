lora_rank=512
gpu_num=4
epoch=3
max_length=512
data_dir=[your_abstraction_data_dir]

input_path=[tuned_model_path]
output_path=[your_output_path]
torchrun --nproc-per-node ${gpu_num} eval_abs_disc.py \
--model_name_or_path ${input_path} --output_dir ${output_path} \
--do_eval --do_predict --report_to none --max_length ${max_length} \
--valid_data_path ${data_dir}/valid.jsonl \
--test_data_path ${data_dir}/test.jsonl \
--evaluation_strategy no --save_strategy no \
--per_device_eval_batch_size 1 --eval_accumulation_steps 1 \
--lora_rank ${lora_rank} --torch_dtype bfloat16 --predict_with_generate