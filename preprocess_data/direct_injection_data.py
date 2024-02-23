import os
import json
import argparse
from collections import defaultdict


def get_eval_event(data_dir):
    cur_eval_event_set = set()
    with open(os.path.join(data_dir, "valid.jsonl")) as fin:
        cur_eval_event_set |= set([json.loads(line)["event"] for line in fin])
    with open(os.path.join(data_dir, "test.jsonl")) as fin:
        cur_eval_event_set |= set([json.loads(line)["event"] for line in fin])
    return cur_eval_event_set


def get_train_data(data_dir, class_flag):
    with open(os.path.join(data_dir, "train.jsonl")) as fin:
        train_data = [json.loads(line) for line in fin]
    # split by relation
    new_train_data = defaultdict(list)
    for d in train_data:
        rel_type = d["type"]
        new_train_data[rel_type].append(d)
    train_data = new_train_data

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_file", type=str)
    parser.add_argument("--abs_disc_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--example_num", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.instruction_file) as fin:
        ins_data = [json.loads(line) for line in fin]

    disc_train_data = get_train_data(args.abs_disc_dir, class_flag=True)

    abs_ins_data = []
    for rel_type in ["noun", "verb", "event"]:
        cur_train_dict = disc_train_data[rel_type]
        label_example_num = args.example_num // len(cur_train_dict)
        for label_type, data_list_by_label in cur_train_dict.items():
            abs_ins_data.extend(data_list_by_label[:label_example_num])
    ins_data = abs_ins_data + ins_data

    output_file = os.path.join(args.output_dir, f"abs_rdm_data_ins{args.example_num}.jsonl")
    with open(output_file, "w") as fout:
        for d in ins_data:
            fout.write(json.dumps(d) + "\n")
