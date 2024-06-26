import logging
import os
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from generation_metric.unify_metrics_api import AutoScorer

from utils import json_load, SupervisedDataset
import json

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from copy import deepcopy
from peft import LoraConfig, TaskType, PeftConfig, PeftModel
from peft import get_peft_model
from utils import is_main_process, init_logger, ds_init_output_dir, format_args
from tqdm import tqdm
from utils import store_generation, smart_tokenizer_and_embedding_resize, DataCollatorForSupervisedDataset
from collections import defaultdict


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train a model "
                          "from scratch."})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, "
                    "the dtype will be automatically derived from the model's weights.",
            "choices": ["auto", "bfloat16", "float16", "float32"]})
    load_from_pretrain: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether load the model from pre-traind or fine-tuned models"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."})
    train_data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    valid_data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."})
    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."})
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    input_col_name: Optional[str] = field(
        default="input",
        metadata={"help": "The name of input column"})
    output_col_name: Optional[str] = field(
        default="output",
        metadata={"help": "The name of output column"})
    lora_rank: int = field(
        default=128, metadata={"help": "the LoRA rank"})
    num_beams: int = field(
        default=1, metadata={"help": "beam search"})
    ref_split_token: str = field(
        default="", metadata={"help": "special token (delimiter) to split references"})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # initialize the output dir
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    if is_main_process(local_rank):
        ds_init_output_dir(training_args)

    # initialize the logger
    with training_args.main_process_first(desc="getting logger"):
        log_level = logging.INFO
        logger = init_logger(training_args, log_level)
        logger.setLevel(log_level)
    logger.info(f"LOCAL RANK of current process: {local_rank}")

    # Log on each process the small summary:
    if is_main_process(local_rank):
        logger.info(
            f"Process rank: {local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(format_args(training_args))
        logger.info(format_args(data_args))
        logger.info(format_args(model_args))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = {}
    if training_args.do_train:
        raw_datasets["train"] = json_load(data_args.train_data_path)
    if training_args.do_eval:
        raw_datasets["valid"] = json_load(data_args.valid_data_path)
    if training_args.do_predict:
        raw_datasets["test"] = json_load(data_args.test_data_path)

    # load peft config if needed
    if model_args.load_from_pretrain:
        model_name_or_path = model_args.model_name_or_path
    else:
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        model_name_or_path = peft_config.base_model_name_or_path

    config = AutoConfig.from_pretrained(
        model_name_or_path)

    model_args.torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype))

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config,
        torch_dtype=model_args.torch_dtype)

    # configure the generation parameters
    gen_kwargs = {
        "max_length": data_args.max_length,
        "min_new_tokens": 1,
        "num_beams": data_args.num_beams,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1,
        "pad_token_id": config.eos_token_id
    }
    if is_main_process(local_rank):
        logger.info(str(gen_kwargs))
    # "min_length": data_args.max_source_length + 1 This is wrong

    model.generation_config.update(**gen_kwargs)

    # initialize the tokenizer and resize the embedding layer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        model_max_length=data_args.max_length,
        padding_side="right"
    )

    # set missing tokens
    if tokenizer.pad_token is None:
        if is_main_process(local_rank):
            logger.info("There is not pad token. Use eos token instead.")
        if config.eos_token_id is None:
            config.eos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token, tokenizer.cls_token = tokenizer.eos_token, tokenizer.eos_token
        config.pad_token_id, config.cls_token_id = config.eos_token_id, config.eos_token_id

        tokenizer.sep_token, tokenizer.mask_token = tokenizer.eos_token, tokenizer.eos_token
        config.sep_token_id, config.mask_token_id = config.eos_token_id, config.eos_token_id

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={},
        tokenizer=tokenizer,
        model=model,
    )

    # load the LoRA config
    if model_args.load_from_pretrain:
        if any(key_word in model_args.model_name_or_path
               for key_word in ["falcon", "Llama-2", "gpt-j", "gpt2", "mpt", "Mistral"]):
            kwargs = {}
        # elif any(key_word in model_args.model_name_or_path for key_word in ["Mistral"]):
        #     kwargs = {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
        else:
            raise ValueError("Model type not included.")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=data_args.lora_rank, lora_alpha=2 * data_args.lora_rank,
                                 lora_dropout=0.1, **kwargs)
        model = get_peft_model(model, peft_config)
    else:
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=training_args.do_train)

    trainable_param, all_param = model.get_nb_trainable_parameters()
    if is_main_process(local_rank):
        logger.info(f"The model is loaded into {model.dtype}")
        param_info = f"trainable params: {trainable_param} || all params: " \
                     f"{all_param} || trainable%: {100 * trainable_param / all_param}"
        logger.info(param_info)
        data_size_str = "raw data size: "
        for key, dataset in raw_datasets.items():
            data_size_str += "{} {},".format(key, len(dataset))
        logger.info(data_size_str)

    tokenized_datasets = {}
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset[: max_train_samples]
        train_dataset = SupervisedDataset(train_dataset, tokenizer, data_args.input_col_name,
                                          data_args.output_col_name, data_args.max_length,
                                          is_eval=False)
        tokenized_datasets["train"] = train_dataset
        if is_main_process(local_rank):
            for index in [0] + random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the train set: {train_dataset[index]}.")
                logger.info(tokenizer.convert_ids_to_tokens(train_dataset[index]["input_ids"]))

    if training_args.do_eval:
        valid_dataset = raw_datasets["valid"]
        if data_args.max_valid_samples is not None:
            max_valid_samples = min(len(valid_dataset), data_args.max_valid_samples)
            valid_dataset = valid_dataset[: max_valid_samples]
        valid_dataset = SupervisedDataset(valid_dataset, tokenizer, data_args.input_col_name,
                                          data_args.output_col_name, data_args.max_length,
                                          is_eval=True)
        tokenized_datasets["valid"] = valid_dataset
        if is_main_process(local_rank):
            for index in random.sample(range(len(valid_dataset)), 3):
                logger.info(f"Sample {index} of the validation set: {valid_dataset[index]}.")
                logger.info(tokenizer.convert_ids_to_tokens(valid_dataset[index]["input_ids"]))

    if training_args.do_predict:
        test_dataset = raw_datasets["test"]
        test_dataset = SupervisedDataset(test_dataset, tokenizer, data_args.input_col_name,
                                         data_args.output_col_name, data_args.max_length,
                                         is_eval=True)
        tokenized_datasets["test"] = test_dataset

    if is_main_process(local_rank):
        data_size_str = "tokenized data size: "
        for key, dataset in tokenized_datasets.items():
            data_size_str += "{} {},".format(key, len(dataset))
        logger.info(data_size_str)

    metric_set = {"bleu", "rouge", "meteor"}
    metric_kwargs = {"bleu": {"max_order": 4}, "rouge": {"use_stemmer": True}, "meteor": {}}
    auto_scorer = AutoScorer(metric_set, reload=False)
    print("finish metric loading")
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # merge LoRA layers if not do_train
    if not training_args.do_train:
        logger.info("not training, then merge LoRA layers")
        model = model.merge_and_unload()

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def compute_metrics(inputs, labels, preds):
        # Replace -100s used for padding as we can't decode them
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        full_decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = labels

        # Some simple post-processing
        decoded_inputs = [d.strip() for d in decoded_inputs]
        full_decoded_preds = [d.strip() for d in full_decoded_preds]
        decoded_labels = [[d.strip() for d in d_list] if isinstance(d_list, list)
                          else [d_list.strip()] for d_list in decoded_labels]
        # remove input
        decoded_preds = []
        assert len(full_decoded_preds) == len(decoded_labels)
        assert len(full_decoded_preds) == len(decoded_inputs)
        for cur_i, cur_p in zip(decoded_inputs, full_decoded_preds):
            decoded_preds.append(cur_p[len(cur_i):])
        # decoded_preds = [[line.strip() for line in d.split("\n") if line.strip()] for d in decoded_preds]
        # decoded_preds = [line[0] for line in decoded_preds]

        result = auto_scorer.compute(inputs=decoded_inputs, preds=decoded_preds,
                                     labels=decoded_labels, metric_kwargs=metric_kwargs)
        for key, value in result.items():
            if isinstance(value, dict):
                result[key] = json.dumps(value)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result, decoded_inputs, decoded_labels, full_decoded_preds, decoded_preds

    # evaluation
    if training_args.do_eval:
        logger.info("*** Validation ***")
        eval_results = trainer.predict(test_dataset=valid_dataset,
                                       metric_key_prefix="valid")
        pred_ids = eval_results.predictions
        input_ids, label_text = [l.tolist() for l in valid_dataset.dataset["input_ids"]], valid_dataset.output_list
        if data_args.ref_split_token != "":
            label_text = [l.split(data_args.ref_split_token) for l in label_text]
        (metrics, decoded_inputs, decoded_labels,
         full_decoded_preds, decoded_preds) = compute_metrics(input_ids, label_text, pred_ids)
        metrics["valid_samples"] = len(valid_dataset)
        trainer.log_metrics("valid", metrics)
        trainer.save_metrics("valid", metrics)
        store_generation(training_args, [input_ids, pred_ids.tolist(), full_decoded_preds,
                                         decoded_inputs, decoded_labels, decoded_preds], split_name="valid")

    if training_args.do_predict:
        logger.info("*** Test ***")
        test_results = trainer.predict(test_dataset=test_dataset,
                                       metric_key_prefix="test")
        pred_ids = test_results.predictions
        input_ids, label_text = [l.tolist() for l in test_dataset.dataset["input_ids"]], test_dataset.output_list
        if data_args.ref_split_token != "":
            label_text = [l.split(data_args.ref_split_token) for l in label_text]
        (metrics, decoded_inputs, decoded_labels,
         full_decoded_preds, decoded_preds) = compute_metrics(input_ids, label_text, pred_ids)
        metrics["test_samples"] = len(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        store_generation(training_args, [input_ids, pred_ids.tolist(), full_decoded_preds,
                                         decoded_inputs, decoded_labels, decoded_preds], split_name="test")

    # write finish file
    if is_main_process(local_rank):
        with open(os.path.join(training_args.output_dir, "checkpoint_finish"), "a") as fout:
            fout.write("training Finished\n")


if __name__ == "__main__":
    main()
