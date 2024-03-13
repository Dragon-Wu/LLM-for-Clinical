import re
import json
import jieba
import string
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from zhon.hanzi import punctuation
from multiprocessing.pool import ThreadPool as Pool

with open("../../data/THUOCL_medical.txt", "r", encoding="utf-8") as f:
    list_word_freq = f.readlines()
list_word_freq = [i.strip().split() for i in list_word_freq]
list_word_freq = [[word, int(freq)] for word, freq in list_word_freq]
for word, freq in list_word_freq:
    jieba.add_word(word.strip(), tag=freq)
    jieba.suggest_freq(word, tune=True)


# function
def clean_string(
    str_processing,
    flag_en=False,
    flag_en_punc=False,
    flag_cn_punc=False,
    flag_num=False,
):
    # remove English character
    if flag_en:
        str_processing = re.sub("[a-zA-Z]", "", str_processing)
    # remove English punctuation
    if flag_en_punc:
        str_processing = re.sub("[{}]".format(string.punctuation), "", str_processing)
    # remove Chinese punctuation
    if flag_cn_punc:
        str_processing = re.sub("[{}]".format(punctuation), "", str_processing)
    # remove Numeric char
    if flag_num:
        str_processing = re.sub("[\d]", "", str_processing)  # [0-9]
    return str_processing


def chinese_in(word):
    for ch in str(word):
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def remove_duplicates(lst):
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            seen[item] = True
            result.append(item)
    return result


# loading question
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

# loading textbooks
with open("../../data/textbook/zh_paragraph/all_books.txt", "r", encoding="utf-8") as f:
    list_corpus_textbooks = f.readlines()


# loading knowledge
def loading_knowledge(path_file):
    with open(path_file, "r", encoding="utf-8") as f:
        data_new = json.load(f)
    list_knowledge = [
        dict_one["Q+A"]
        + dict_one["Q+B"]
        + dict_one["Q+C"]
        + dict_one["Q+D"]
        + dict_one["Q+E"]
        for dict_one in data_new
    ]
    list_knowledge = remove_duplicates(flatten(list_knowledge))
    return list_knowledge


# merge and filtering
list_corpus = remove_duplicates(list_corpus_textbooks)
print(f"Loading {len(list_corpus)} knowledge")

list_corpus_fined = []
for line_one in list_corpus:
    line_one_cleaned = clean_string(
        line_one, flag_en=True, flag_en_punc=True, flag_cn_punc=True, flag_num=True
    )
    # must contain 10 word and contain Chinese
    if len(line_one_cleaned) > 10 and chinese_in(line_one_cleaned):
        list_corpus_fined.append(line_one.strip())
print(f"Filtered knowledge: {len(list_corpus_fined)} items")

# BM Bank
corpus_knowledge = [list(jieba.cut(doc)) for doc in list_corpus_fined]
bm25_knowledge = BM25Okapi(corpus_knowledge)


# find knowledge for each question+option
def find_knowledge_question_option(
    bm, bm_corpus, question, option, top_n, flag_print=None
):
    # Only question: if only number after removing punctuation
    if clean_string(option, flag_en_punc=True, flag_cn_punc=True).isnumeric():
        query = question
    # repeat option to same length of question
    else:
        query = question + " " + option
    tokenized_query = list(jieba.cut(query))
    if flag_print:
        print(tokenized_query)
    list_similar = bm.get_top_n(tokenized_query, bm_corpus, n=top_n)
    list_similar = ["".join(list_str) for list_str in list_similar]
    return list_similar


# Run
def get_kg_question_option(dict_test):
    question = dict_test["Question"]
    # set_knowledge = set()
    for option in ["A", "B", "C", "D", "E"]:
        option_name = dict_test[option]
        list_knowledge_question_option = find_knowledge_question_option(
            bm25_knowledge, corpus_knowledge, question, option_name, top_n=5
        )
        dict_test[f"Q+{option}"] = list_knowledge_question_option
        # dict_test[f"Q+{option}"] = get_knowledge(set_knowledge, list_knowledge_question_option)
    return dict_test


# multi-thread run
pool = Pool(12)
list_dict_test_new = pool.map(get_kg_question_option, tqdm(list_dict_test))
pool.close()
pool.join()

# save
with open("data_knowledge_enhancement.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(list_dict_test_new, indent=2, ensure_ascii=False))
