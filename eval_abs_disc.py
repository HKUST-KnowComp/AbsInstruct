import logging
import os
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn import metrics as skmetrics

from utils import json_load, SupervisedDataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import PeftConfig, PeftModel
from utils import is_main_process, init_logger, ds_init_output_dir, format_args
from utils import store_generation, smart_tokenizer_and_embedding_resize, DataCollatorForSupervisedDataset
from utils import parse_label


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
    if training_args.do_eval:
        raw_datasets["valid"] = json_load(data_args.valid_data_path)
    if training_args.do_predict:
        raw_datasets["test"] = json_load(data_args.test_data_path)

    # load peft config if needed
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
    model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=training_args.do_train)
    logger.info("not training, then merge LoRA layers")
    model = model.merge_and_unload()

    if is_main_process(local_rank):
        data_size_str = "raw data size: "
        for key, dataset in raw_datasets.items():
            data_size_str += "{} {},".format(key, len(dataset))
        logger.info(data_size_str)

    tokenized_datasets = {}
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
        if data_args.max_test_samples is not None:
            max_test_samples = min(len(test_dataset), data_args.max_test_samples)
            test_dataset = test_dataset[: max_test_samples]
        test_dataset = SupervisedDataset(test_dataset, tokenizer, data_args.input_col_name,
                                         data_args.output_col_name, data_args.max_length,
                                         is_eval=True)
        tokenized_datasets["test"] = test_dataset

    if is_main_process(local_rank):
        data_size_str = "tokenized data size: "
        for key, dataset in tokenized_datasets.items():
            data_size_str += "{} {},".format(key, len(dataset))
        logger.info(data_size_str)

    metric_fns = [('accuracy', skmetrics.accuracy_score), ('f1', skmetrics.f1_score),
                  ('precision', skmetrics.precision_score), ('recall', skmetrics.recall_score),
                  ('ma-f1', skmetrics.f1_score)]

    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    def compute_metrics(inputs, labels, preds):
        # Replace -100s used for padding as we can't decode them
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        full_decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = labels

        # Some simple post-processing
        decoded_inputs = [d.strip() for d in decoded_inputs]
        full_decoded_preds = [d.strip() for d in full_decoded_preds]
        decoded_labels = [d.strip() for d in decoded_labels]
        # remove input
        decoded_preds = []
        assert len(full_decoded_preds) == len(decoded_labels)
        assert len(full_decoded_preds) == len(decoded_inputs)
        for cur_i, cur_p in zip(decoded_inputs, full_decoded_preds):
            decoded_preds.append(cur_p[len(cur_i):])
        # convert to labels
        num_preds = [parse_label(d) for d in decoded_preds]
        num_labels = [parse_label(d) for d in decoded_labels]
        binary_num_preds = [d if d in {0, 1} else 0 for d in num_preds]

        results = {}
        for name, fn in metric_fns:
            if name == 'ma-f1':
                results[name] = fn(num_labels, binary_num_preds, average="macro")
            else:
                results[name] = fn(num_labels, binary_num_preds)
        # concatenate numbers back to text
        decoded_preds = [dp + f"|num:{num_p}" for dp, num_p in zip(decoded_preds, num_preds)]
        decoded_labels = [dl + f"|num:{num_l}" for dl, num_l in zip(decoded_labels, num_labels)]
        return results, decoded_inputs, decoded_labels, full_decoded_preds, decoded_preds

    # evaluation
    if training_args.do_eval:
        logger.info("*** Validation ***")
        eval_results = trainer.predict(test_dataset=valid_dataset,
                                       metric_key_prefix="valid")
        pred_ids = eval_results.predictions
        input_ids, label_text = [l.tolist() for l in valid_dataset.dataset["input_ids"]], valid_dataset.output_list
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
