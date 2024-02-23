import os
import json
import argparse
from collections import defaultdict, Counter
from cot_disc_format import cot_prompt_template
from abs_disc_format import prompt_template
from rouge_score import rouge_scorer
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATA_TYPE_LIST = ["noun", "verb", "event"]


def get_train_data(data_dir, class_flag):
    train_data = defaultdict(list)
    for data_type in DATA_TYPE_LIST:
        with open(os.path.join(data_dir, f"{data_type}_train.json")) as fin:
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


def get_max_rouge_score(cur_scorer, cur_rationale_text, cur_ref_list):
    rouge_list = []
    for ref in cur_ref_list:
        score = cur_scorer.score(ref, cur_rationale_text)
        for key, value in score.items():
            value = getattr(value, "fmeasure")
            rouge_list.append(value)
    max_rouge = max(rouge_list) if rouge_list else 0
    return max_rouge


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_file", type=str)
    parser.add_argument("--abs_disc_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--example_num", type=int)
    parser.add_argument("--ppl_type", type=str,
                        choices=["low", "high", "mix"])
    parser.add_argument("--no_step_by_step", action="store_true")
    args = parser.parse_args()

    print(args)
    with open(args.instruction_file) as fin:
        ins_data = [json.loads(line) for line in fin]

    disc_train_data = get_train_data(args.abs_disc_dir, class_flag=True)
    for data_type in DATA_TYPE_LIST:
        for label, data_list in disc_train_data[data_type].items():
            if len(data_list) > 500:
                disc_train_data[data_type][label] = data_list[:500]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    print("select by ppl")
    abs_ins_data, diverse_filter_count = [], 0
    for rel_type in DATA_TYPE_LIST:
        cur_train_dict = disc_train_data[rel_type]
        label_example_num = args.example_num // len(cur_train_dict)
        for label, cur_data_list in cur_train_dict.items():
            assert label_example_num <= len(cur_data_list), "no enough data"
            if args.ppl_type == "low":
                cur_data_list = sorted(cur_data_list, key=lambda x: x["part_ppl"], reverse=False)
            elif args.ppl_type == "high":
                cur_data_list = sorted(cur_data_list, key=lambda x: x["part_ppl"], reverse=True)
            elif args.ppl_type == "mix":
                cur_data_list = sorted(cur_data_list, key=lambda x: x["part_ppl"], reverse=False)
                mix_example_num = label_example_num // 2
                new_data_list = cur_data_list[:label_example_num - mix_example_num]
                new_data_list += cur_data_list[-mix_example_num:]
                cur_data_list = new_data_list
            all_instruction_tokens, cur_abs_ins_data = [], []
            for d in cur_data_list:
                event_text, concept_text = d["event"], d["concept"]
                start_idx, end_idx = event_text.index("<"), event_text.index(">")
                instance_text = event_text[start_idx + 1: end_idx]
                assert instance_text in event_text
                if not args.no_step_by_step:
                    instruction = cot_prompt_template[rel_type]["instruction"]
                    input_template = cot_prompt_template[rel_type]["input"]
                    output_template = cot_prompt_template[rel_type]["output"]
                    if rel_type in {"noun", "verb"}:
                        input_text = input_template.format(event_text, concept_text, instance_text)
                    else:
                        input_text = input_template.format(concept_text, event_text)

                    step1_arg = d["chatgpt-gen"][0] + " Meanwhile, " + d["chatgpt-gen"][1]

                    new_instruction_tokens = scorer._tokenizer.tokenize(step1_arg)
                    rouge_score_list = [rouge_scorer._score_lcs(new_instruction_tokens, ref_tokens)
                                        for ref_tokens in all_instruction_tokens]
                    rouge_scores = [score.fmeasure for score in rouge_score_list]
                    if max(rouge_scores + [0]) > 0.7:
                        print(step1_arg)
                        diverse_filter_count += 1
                        continue
                    all_instruction_tokens.append(new_instruction_tokens)

                    if d["label"] == 1:
                        step2_arg = "Yes, the meaning of \"{}\" encompasses \"{}.\"".format(concept_text, instance_text)
                        output_text = output_template.format(step1_arg, step2_arg)
                    else:
                        step2_arg = "No, the meaning of \"{}\" does not " \
                                    "encompass \"{}.\"".format(concept_text, instance_text)
                        output_text = output_template.format(step1_arg, step2_arg)
                    new_d = {"id": d["id"], "instruction": instruction, "input": input_text,
                             "output": output_text, "part_ppl": d["part_ppl"], "model": d["model"]}
                    cur_abs_ins_data.append(new_d)

                # abs data without cot
                instruction = prompt_template[rel_type]["instruction"]
                input_template = prompt_template[rel_type]["input"]
                if rel_type in {"noun", "verb"}:
                    input_text = input_template.format(event_text, concept_text, instance_text)
                else:
                    input_text = input_template.format(concept_text, event_text)
                output_text = "Yes" if d["label"] else "No"
                new_d = {"id": d["id"], "instruction": instruction, "input": input_text,
                         "output": output_text, "part_ppl": d["part_ppl"], "model": d["model"]}
                cur_abs_ins_data.append(new_d)
                if len(cur_abs_ins_data) == 2 * label_example_num:
                    abs_ins_data.extend(cur_abs_ins_data)
                    break

    print(f"{diverse_filter_count} examples are filtered for rouge-L")

    ins_data = abs_ins_data + ins_data
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(args.output_dir, f"abs_rdm_data_ins{args.example_num}{args.ppl_type}.jsonl")
    with open(output_file, "w") as fout:
        for d in ins_data:
            fout.write(json.dumps(d) + "\n")
