import io

import tqdm
import logging
import os
import sys
import datasets
import transformers
from dataclasses import asdict
import shutil
from torch import nn
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from dataclasses import dataclass, field
from sklearn.metrics import precision_recall_curve
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, Optional, Sequence
import re
import random

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further "
                    "context. Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:",
    "prompt_no_input": "Below is an instruction that describes a task. "
                       "Write a response that appropriately completes the request.\n\n"
                       "### Instruction:\n{}\n\n### Response:",
}

ICL_PROMPT_DICT = {
    "prompt_input_ins": "Below is an instruction that describes a task, paired with an input that provides further "
                    "context. Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{}\n\n",
    "prompt_input": "### Input:\n{}\n\n### Response:",
}


def init_logger(training_args, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # init a formatter to add date information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # init a file handler and a stream handler
    fh = logging.FileHandler(os.path.join(training_args.output_dir, "train.log"), encoding="utf-8", mode="a")
    fh.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    # set formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add those handlers to the root logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # the logger level of huggingface packages
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()

    return logger


def format_args(args):
    args_as_dict = asdict(args)
    # args_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in args_as_dict.items()}
    attrs_as_str = [f"{k}={v}," for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"


def ds_init_output_dir(training_args):
    if os.path.exists(training_args.output_dir):
        if os.path.exists(os.path.join(training_args.output_dir, "checkpoint_finish")) > 0:
            raise ValueError(
                "training/inference process in dir {} is finished, plz clear it manually.".format(training_args.output_dir))
        if training_args.do_train:
            shutil.rmtree(training_args.output_dir, ignore_errors=True)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    os.system("touch {}".format(os.path.join(training_args.output_dir, "train.log")))


def revise_mnli_models(model_name_or_path, mnli_model, neutral_id, entail_id):
    if "bart" in model_name_or_path:
        head = mnli_model.classification_head
        linear = head.out_proj  # n x 3
    elif "roberta" in model_name_or_path:
        head = mnli_model.classifier
        linear = head.out_proj
    elif "deberta" in model_name_or_path:
        linear = mnli_model.classifier
    else:
        raise ValueError

    # copy weight and bias
    hidden_size = linear.weight.shape[-1]
    new_linear = nn.Linear(hidden_size, 2)  # n x 2
    with torch.no_grad():
        linear_weight = torch.stack([linear.weight[neutral_id, :], linear.weight[entail_id, :]], dim=0)
        linear_bias = torch.stack([linear.bias[neutral_id], linear.bias[entail_id]])
        new_linear.weight.data = linear_weight
        new_linear.bias.new_data_list = linear_bias

    if "bart" in model_name_or_path:
        mnli_model.classification_head.out_proj = new_linear
    elif "roberta" in model_name_or_path:
        mnli_model.classifier.out_proj = new_linear
    elif "deberta" in model_name_or_path:
        mnli_model.classifier = new_linear

    # change config
    mnli_model.config.num_labels = 2

    if hasattr(mnli_model, "num_labels"):
        mnli_model.num_labels = 2

    mnli_model.eval()

    return mnli_model


def is_main_process(local_rank):
    return local_rank == 0 or local_rank == -1


def average_precision_score(y_true, y_score, pos_label=1):
    precision, recall, _ = precision_recall_curve(
        y_true, y_score, pos_label=pos_label
    )
    print(len(precision), precision)
    print(len(recall), recall)
    recall_diff, precision = np.diff(recall), np.array(precision)[:-1]
    high_precision_mask = precision > 0.5
    print(len(high_precision_mask), high_precision_mask)
    recall_diff, precision = recall_diff[high_precision_mask], precision[high_precision_mask]
    # print(len(recall_diff), recall_diff)
    # print(len(precision), precision)
    return -np.sum(recall_diff * precision)


def store_generation(training_args, text_list, split_name):
    with open(os.path.join(training_args.output_dir, "{}.jsonl".format(split_name)), "w") as fout:
        for ri, rp, tp, i, l, p in tqdm(zip(*text_list), "output generations"):
            fout.write(json.dumps({"input": i, "label": l, "pred": p,
                                   "raw_input": ri, "raw_pred": rp, "text_pred": tp}) + "\n")


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.new_data_list
        output_embeddings = model.get_output_embeddings().weight.new_data_list

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, tokenizer, input_col_name, output_col_name,
                 max_length, is_eval, prompt_dict=None):
        super(SupervisedDataset, self).__init__()
        logging.info("Loading data...")
        self.tokenizer, self.is_eval = tokenizer, is_eval
        self.input_col_name, self.output_col_name = input_col_name, output_col_name
        self.max_length = max_length
        self.output_list = [d[self.output_col_name] for d in dataset]
        if prompt_dict is not None:
            self.prompt_dict = prompt_dict
        else:
            self.prompt_dict = PROMPT_DICT
        self.dataset = self.tokenize_function(dataset)
        print(self.prompt_dict)

        logging.info("Formatting inputs...")

    def tokenize_function(self, list_data_dict):
        prompt_input, prompt_no_input = self.prompt_dict["prompt_input"], self.prompt_dict["prompt_no_input"]
        source_list, example_list = [], []
        for idx in tqdm(range(len(list_data_dict)), "verbalizing"):
            instruction = list_data_dict[idx]["instruction"]
            input_text = list_data_dict[idx][self.input_col_name]
            if input_text != "":
                source = prompt_input.format(instruction, input_text)
            else:
                source = prompt_no_input.format(instruction)
            source_list.append(source)
            # init template
            if not self.is_eval:
                target = list_data_dict[idx][self.output_col_name]
                target = f"{target}{self.tokenizer.eos_token}"
                example = source + " " + target
                example_list.append(example)

        if self.is_eval:
            source_list = self.tokenizer(source_list,
                                         padding="do_not_pad", max_length=self.max_length,
                                         truncation=True)
            for i in range(len(source_list["input_ids"])):
                source_list["input_ids"][i] = torch.tensor(source_list["input_ids"][i], dtype=torch.int64)
                source_list["attention_mask"][i] = torch.tensor(source_list["attention_mask"][i], dtype=torch.int64)
            return source_list
        else:
            example_list = self.tokenizer.batch_encode_plus(example_list, padding='do_not_pad',
                                                            max_length=self.max_length,
                                                            truncation=True)
            source_list = self.tokenizer.batch_encode_plus(source_list, padding='do_not_pad',
                                                           max_length=self.max_length,
                                                           truncation=True)

            example_list["labels"] = deepcopy(example_list["input_ids"])  # assigning the value is just shallow copy
            for i in range(len(example_list['input_ids'])):
                # example, source = example_list[i], source_list[i]
                source_list["input_ids"][i] = torch.tensor(source_list["input_ids"][i], dtype=torch.int64)
                source_list["attention_mask"][i] = torch.tensor(source_list["attention_mask"][i], dtype=torch.int64)

                example_list["input_ids"][i] = torch.tensor(example_list["input_ids"][i], dtype=torch.int64)
                example_list["attention_mask"][i] = torch.tensor(example_list["attention_mask"][i], dtype=torch.int64)
                example_list["labels"][i] = torch.tensor(example_list["labels"][i], dtype=torch.int64)
                # for computing loss on tail,
                # reset label tokens from head and relation by -100, so GPT2LMHealModel will not compute loss on that
                source_len = source_list["attention_mask"][i].sum().item()
                example_list["labels"][i][:source_len] = IGNORE_INDEX
            return example_list

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return_dict = {"input_ids": self.dataset["input_ids"][i],
                       "attention_mask": self.dataset["attention_mask"][i]}
        if not self.is_eval:
            return_dict["labels"] = self.dataset["labels"][i]
        return return_dict

prompt_template = {
    "noun": {
        "instruction": "Identify the hypernym of a specific noun and provide a \"Yes\" or \"No\" response. "
                       "Hypernyms are words with a broad meaning, which more specific words fall under.",
        "input":
            "In the sentence \"{},\" does the meaning of \"{}\" encompass \"{}?\""},
    "verb": {
        "instruction": "Identify the hypernym of a specific verb and provide a \"Yes\" or \"No\" response. "
                       "Hypernyms are words with a broad meaning, which more specific words fall under.",
        "input":
            "In the sentence \"{},\" does the meaning of \"{}\" encompass \"{}?\""},
    "event": {
        "instruction": "Identify abstract descriptions of specific sentences, and "
                       "provide a \"Yes\" or \"No\" response.",
        "input":
            "Can we consider \"{}\" as an abstract description of the sentence \"{}?\""}}
class ICLDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, tokenizer, input_col_name, output_col_name,
                 max_length, exemplar_list, is_eval):
        super(ICLDataset, self).__init__()
        logging.info("Loading data...")
        self.tokenizer, self.is_eval = tokenizer, is_eval
        self.input_col_name, self.output_col_name = input_col_name, output_col_name
        self.max_length = max_length
        self.output_list = [d[self.output_col_name] for d in dataset]

        self.exemplar_list = exemplar_list
        self.dataset = self.tokenize_function(dataset)

        logging.info("Formatting inputs...")


    def tokenize_function(self, list_data_dict):
        prompt_input = ICL_PROMPT_DICT["prompt_input"]
        source_list, example_list = [], []
        for idx in tqdm(range(len(list_data_dict)), "verbalizing"):
            input_text = list_data_dict[idx][self.input_col_name]
            source = prompt_input.format(input_text)
            source = self.exemplar_list + source
            source_list.append(source)
            # init template
            if not self.is_eval:
                target = list_data_dict[idx][self.output_col_name]
                target = f"{target}{self.tokenizer.eos_token}"
                example = source + " " + target
                example_list.append(example)

        if self.is_eval:
            source_list = self.tokenizer(source_list,
                                         padding="do_not_pad", max_length=self.max_length,
                                         truncation=True)
            for i in range(len(source_list["input_ids"])):
                source_list["input_ids"][i] = torch.tensor(source_list["input_ids"][i], dtype=torch.int64)
                source_list["attention_mask"][i] = torch.tensor(source_list["attention_mask"][i], dtype=torch.int64)
            return source_list
        else:
            return example_list

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return_dict = {"input_ids": self.dataset["input_ids"][i],
                       "attention_mask": self.dataset["attention_mask"][i]}
        if not self.is_eval:
            return_dict["labels"] = self.dataset["labels"][i]
        return return_dict



class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instance_list: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instance_list]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = [instance["attention_mask"] for instance in instance_list]
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0)
        element_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        # reset [PAD] token by -100, so GPT2LMHealModel will not compute loss on that
        # but not the first [PAD] token, which is [EOS]
        if "labels" in instance_list[0]:
            labels = [instance["labels"] for instance in instance_list]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            element_dict["labels"] = labels

        return element_dict


def json_load(f, mode="r"):
    """Load a .json file into a dictionary."""
    with open(f, mode=mode) as fin:
        list_data_dict = [json.loads(line) for line in fin]
    return list_data_dict


word_map = {"yes": 1, "no": 0}
def parse_label(text):
    text = [line.strip() for line in text.lower().split("\n") if line.strip()]
    text = "\n".join(text[::-1])
    if text in word_map:
        return word_map[text]
    index_map = {"yes": 1e5, "no": 1e5}
    for word in word_map:
        if word in text:
            index_map[word] = text.index(word)
    if index_map["yes"] == 1e5 and index_map["no"] == 1e5:
        return -1
    elif index_map["yes"] < index_map["no"]:
        return 1
    elif index_map["yes"] > index_map["no"]:
        return 0
    else:
        return -1

