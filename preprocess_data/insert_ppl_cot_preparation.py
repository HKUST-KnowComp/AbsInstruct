import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
from cot_disc_format import cot_prompt_template
from abs_disc_format import prompt_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cross_entropy

DATA_TYPE_LIST = ["noun", "verb", "event"]


def get_train_data(data_dir, class_flag):
    train_data = defaultdict(list)
    for data_type in DATA_TYPE_LIST:
        with open(os.path.join(data_dir, f"{data_type}_dataset/train_pred.json")) as fin:
            cur_train_data = [json.loads(line) for line in fin]
            train_data[data_type] = cur_train_data

    # if classification, split by labels
    if class_flag:
        new_train_data = {}
        for rel_type, data_list in train_data.items():
            data_list_by_label = defaultdict(list)
            for d in data_list:
                data_list_by_label[d["label"]].append(d)
            new_train_data[rel_type] = data_list_by_label
        train_data = new_train_data
    return train_data


def get_most_frequent_list(my_list):
    occurence_count = Counter(my_list)
    return occurence_count.most_common(1)[0][0]


def prepare_test_sent(data_dict_list, rel_type, ppl_type, model_name):
    example_list, offset_list = [], []
    for d in data_dict_list:
        event_text, concept_text = d["event"], d["concept"]
        start_idx, end_idx = event_text.index("<"), event_text.index(">")
        instance_text = event_text[start_idx + 1: end_idx]
        instruction = cot_prompt_template[rel_type]["instruction"]
        input_template = cot_prompt_template[rel_type]["input"]
        output_template = cot_prompt_template[rel_type]["output"]
        if rel_type in {"noun", "verb"}:
            input_text = input_template.format(event_text, concept_text, instance_text)
        else:
            input_text = input_template.format(concept_text, event_text)
        input_text = instruction + " " + input_text

        step1_arg = d["chatgpt-gen"][0] + " Meanwhile, " + d["chatgpt-gen"][1]
        if d["label"] == 1:
            step2_arg = "Yes, the meaning of \"{}\" encompasses \"{}.\"".format(concept_text, instance_text)
            output_text = output_template.format(step1_arg, step2_arg)
        else:
            step2_arg = "No, the meaning of \"{}\" does not " \
                        "encompass \"{}.\"".format(concept_text, instance_text)
            output_text = output_template.format(step1_arg, step2_arg)

        if ppl_type == "cot":
            offset = len(input_text)
        elif ppl_type == "ans":
            offset = len(input_text) + 1 + output_text.index("Step 2:")
        elif ppl_type == "full":
            offset = 0
        else:
            raise ValueError("Wrong value")
        if "mpt" in model_name:
            offset += 1
        offset_list.append(offset)
        example_list.append(input_text + " " + output_text)

    return example_list, offset_list


def get_loss(inputs, partial_idx):
    outputs = model(**inputs)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    cur_batch_size = len(inputs["attention_mask"])

    shift_logits = outputs[0][..., :-1, :].contiguous().view(-1, outputs[0].size(-1))
    shift_labels = ids[..., 1:].contiguous().view(-1)

    losses = cross_entropy(shift_logits, shift_labels,
                           ignore_index=tokenizer.pad_token_id, reduction="none").view(ids.size(0), -1)
    partial_loss_list = [torch.sum(losses[idx, partial_idx[idx] - 1:]) /
                         torch.sum(mask[idx, partial_idx[idx] - 1:]) for idx in range(0, cur_batch_size)]
    partial_score_list = [pl.item() for pl in partial_loss_list]

    return partial_score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs_disc_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--ppl_type", type=str, choices=["ans", "cot", "full"])
    parser.add_argument("--no_filter", action="store_true")
    args = parser.parse_args()

    print(args)

    disc_train_data = get_train_data(args.abs_disc_dir, class_flag=True)

    # filter by label
    if not args.no_filter:
        filtered_data_list = []
        print("filter by label")
        for data_type in DATA_TYPE_LIST:
            for label, data_list in disc_train_data[data_type].items():
                new_data_list = []
                for d in data_list:
                    if get_most_frequent_list(d["chatgpt-label"]) == d["label"]:
                        new_data_list.append(d)
                    else:
                        filtered_data_list.append(d)
                print(data_type, label, f"kept: {len(new_data_list)}, removed: {len(data_list) - len(new_data_list)}")
                disc_train_data[data_type][label] = new_data_list
        for d in filtered_data_list:
            print(d)

    # filter by parsing
    print("filter by parsing")
    for data_type in DATA_TYPE_LIST:
        for label, data_list in disc_train_data[data_type].items():
            new_data_list = []
            for d in data_list:
                rationale_text = d["chatgpt-gen"][0].split("\n")
                rationale_text = [t.strip() for t in rationale_text if t.strip()]
                if len(rationale_text) != 3:
                    continue
                if not rationale_text[0].startswith("Step 1:") or not rationale_text[1].startswith("Step 2:") \
                        or not rationale_text[2].startswith("Step 3:"):
                    continue
                else:
                    rationale_text[0] = rationale_text[0][len("Step 1:"):].strip()
                    rationale_text[1] = rationale_text[1][len("Step 2:"):].strip()
                    rationale_text[2] = rationale_text[2][len("Step 3:"):].strip()
                d["chatgpt-gen"] = rationale_text
                new_data_list.append(d)
            print(data_type, label, f"kept: {len(new_data_list)}, removed: {len(data_list) - len(new_data_list)}")
            disc_train_data[data_type][label] = new_data_list

    # filter by key word
    if not args.no_filter:
        filtered_data_list = []
        print("filter by keyword")
        for data_type in DATA_TYPE_LIST:
            for label, data_list in disc_train_data[data_type].items():
                new_data_list = []
                for d in data_list:
                    event_text, concept_text = d["event"], d["concept"]
                    start_idx, end_idx = event_text.index("<"), event_text.index(">")
                    instance_text = event_text[start_idx + 1: end_idx]
                    rationale_text = d["chatgpt-gen"]
                    instance_text, concept_text = instance_text.lower(), concept_text.lower()
                    if instance_text in rationale_text[0].lower() and concept_text in rationale_text[1].lower():
                        new_data_list.append(d)
                    else:
                        filtered_data_list.append(d)
                print(data_type, label, f"kept: {len(new_data_list)}, removed: {len(data_list) - len(new_data_list)}")
                disc_train_data[data_type][label] = new_data_list
        # for d in filtered_data_list:
        #     print(d)
    # add ppl
    print("adding ppl")
    print("start load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print("finish load tokenizer")
    if tokenizer.pad_token is None:
        print("There is not pad token. Use eos token instead.")
        tokenizer.pad_token, tokenizer.cls_token = tokenizer.eos_token, tokenizer.eos_token
        tokenizer.sep_token, tokenizer.mask_token = tokenizer.eos_token, tokenizer.eos_token
    # AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()

    for data_type in DATA_TYPE_LIST:
        for label in [0, 1]:
            data_list = disc_train_data[data_type][label]
            example_list, offset_list = prepare_test_sent(data_list, data_type, args.ppl_type, args.model_name_or_path)
            partial_score_list = []
            with torch.no_grad():
                for i in tqdm(range(0, len(example_list), args.batch_size)):
                    cur_text_list = example_list[i: i + args.batch_size]
                    cur_offset_list = offset_list[i: i + args.batch_size]
                    inputs = tokenizer.batch_encode_plus(cur_text_list, padding="longest",
                                                         return_tensors="pt", return_offsets_mapping=True)
                    token_offset_list = []
                    for j in range(0, len(inputs["offset_mapping"])):
                        for k in range(len(inputs["offset_mapping"][j])):
                            if cur_offset_list[j] == inputs["offset_mapping"][j][k][0] and \
                                    inputs["offset_mapping"][j][k][0] != inputs["offset_mapping"][j][k][1]:
                                token_offset_list.append(max(k, 1))  # at lest the second token
                                break

                    assert len(token_offset_list) == len(inputs["offset_mapping"])
                    inputs.pop("offset_mapping")
                    cur_score_list = get_loss(inputs.to("cuda"), token_offset_list)
                    partial_score_list.extend(cur_score_list)
            assert len(partial_score_list) == len(data_list)
            for ppl_score, d in zip(partial_score_list, data_list):
                d["part_ppl"] = ppl_score
                d["model"] = args.model_name_or_path

    # reformat by data_type
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for data_type in DATA_TYPE_LIST:
        type_data_list = []
        for label in [0, 1]:
            data_list = disc_train_data[data_type][label]
            type_data_list.extend(data_list)
        output_file = os.path.join(args.output_dir, f"{data_type}_train.json")
        with open(output_file, "w") as fout:
            for d in type_data_list:
                fout.write(json.dumps(d) + "\n")
