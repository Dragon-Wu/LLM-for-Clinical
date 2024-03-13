import re
import json
import jieba
import numpy as np
from tqdm import tqdm
from thefuzz import fuzz
from rank_bm25 import BM25Okapi

with open("../../data/THUOCL_medical.txt", "r", encoding="utf-8") as f:
    list_word_freq = f.readlines()
list_word_freq = [i.strip().split() for i in list_word_freq]
list_word_freq = [[word, int(freq)] for word, freq in list_word_freq]
for word, freq in list_word_freq:
    jieba.add_word(word.strip(), tag=freq)
    jieba.suggest_freq(word, tune=True)


def get_similar_idx(option, n_top, model):
    tokenized_option = list(jieba.cut(option))
    doc_scores = model.get_scores(tokenized_option)
    list_sample_idx = np.argsort(doc_scores, axis=0)[-n_top:][::-1]
    return list_sample_idx


def get_similar_sample(option, n_top, model, list_sample):
    tokenized_option = list(jieba.cut(option))
    doc_scores = model.get_scores(tokenized_option)
    list_sample_idx = np.argsort(doc_scores, axis=0)[-n_top:][::-1]
    list_sample_similar = [list_sample[idx] for idx in list_sample_idx]
    return list_sample_similar


# loading data
path_file_data = "../../data/MedQA/Mainland/test.jsonl"
list_dict_test = []
with open(path_file_data, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        data["ID"] = idx
        data["A"], data["B"], data["C"], data["D"], data["E"] = (
            data["options"]["A"],
            data["options"]["B"],
            data["options"]["C"],
            data["options"]["D"],
            data["options"]["E"],
        )
        del data["options"]
        list_dict_test.append(data)

# loading question bank
list_path_file_qb = [
    "../../data/MedQA/Mainland/train.jsonl",
    "../../data/MedQA/Mainland/dev.jsonl",
]
list_dict_sample_all = []
for path_file_qb in list_path_file_qb:
    with open(path_file_qb, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            data["ID"] = idx
            data["A"], data["B"], data["C"], data["D"], data["E"] = (
                data["options"]["A"],
                data["options"]["B"],
                data["options"]["C"],
                data["options"]["D"],
                data["options"]["E"],
            )
            del data["options"]
            list_dict_sample_all.append(data)
print(f"Loading {len(list_dict_sample_all)} samples")


# write a function to remove the duplicate dict of list_sample_all and keep the order
def remove_duplicate_dict(list_sample_all):
    list_duplicate = []
    list_sample_all_new = [list_sample_all[0]]
    for dict_sample in tqdm(list_sample_all[1:]):
        flag_in = False
        str_option_sample = (
            dict_sample["A"]
            + " "
            + dict_sample["B"]
            + " "
            + dict_sample["C"]
            + " "
            + dict_sample["D"]
            + " "
            + dict_sample["E"]
        )
        for dict_in in list_sample_all_new:
            str_option_in = (
                dict_in["A"]
                + " "
                + dict_in["B"]
                + " "
                + dict_in["C"]
                + " "
                + dict_in["D"]
                + " "
                + dict_in["E"]
            )
            if (
                fuzz.ratio(dict_in["question"], dict_sample["question"]) > 50
                and fuzz.ratio(str_option_sample, str_option_in) > 65
            ):
                flag_in = True
                list_duplicate.append(
                    [
                        dict_in["question"],
                        str_option_in,
                        dict_sample["question"],
                        str_option_sample,
                    ]
                )
                break
        if not flag_in:
            list_sample_all_new.append(dict_sample)
    return list_sample_all_new, list_duplicate


print(f"The number of sample before removing duplicate: {len(list_dict_sample_all)}")
list_dict_sample_all, list_duplicate = remove_duplicate_dict(list_dict_sample_all)
print(f"The number of sample after removing duplicate: {len(list_dict_sample_all)}")

list_dict_sample = []
for idx, dict_sample_all in enumerate(tqdm(list_dict_sample_all)):
    question, answer = dict_sample_all["question"], dict_sample_all["answer"]
    a, b, c, d, e = (
        dict_sample_all["A"],
        dict_sample_all["B"],
        dict_sample_all["C"],
        dict_sample_all["D"],
        dict_sample_all["E"],
    )
    question = re.sub(r"\s*（\s*）。*$", "", question)
    question = re.sub(r"\s*\(\s*\)。*$", "", question)
    dict_sample = {
        "ID": dict_sample_all["ID"],
        "question": question,
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "E": e,
        "answer": answer,
    }
    input = (
        f"问题：{question}: (A){a}, (B){b}, (C){c}, (D){d}, (E){e}\n答案：{answer}\n"
    )
    dict_sample["Input"] = input
    list_dict_sample.append(dict_sample)


corpus_question = [
    dict_one["question"]
    + " "
    + dict_one["A"]
    + " "
    + dict_one["B"]
    + " "
    + dict_one["C"]
    + " "
    + dict_one["D"]
    + " "
    + dict_one["E"]
    for dict_one in list_dict_sample
]
tokenized_corpus_question = [list(jieba.cut(doc)) for doc in corpus_question]
bm25_question = BM25Okapi(tokenized_corpus_question)
print(f"We construct a Question Bank with {len(corpus_question)} questions")

num_similar = 10


def main():
    for idx_testing, dict_test in enumerate(tqdm(list_dict_test[:50])):
        question_and_options = (
            dict_test["question"]
            + " "
            + dict_test["A"]
            + " "
            + dict_test["B"]
            + " "
            + dict_test["C"]
            + " "
            + dict_test["D"]
            + " "
            + dict_test["E"]
        )
        list_similar_sample = get_similar_sample(
            question_and_options,
            n_top=num_similar,
            model=bm25_question,
            list_sample=list_dict_sample,
        )
        dict_test["list_similar_sample"] = list_similar_sample

    with open(f"data_fewshot_enhancement.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(list_dict_test, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
