# coding=utf-8
import sys
import json
import time
from tqdm import tqdm
import multiprocessing

import openai

# config of model
model = "gpt-4-0613"
cost_input_1k = 0.03
cost_output_1k = 0.06
temperature = 0
top_p = 1
frequency_penalty = 0
presence_penalty = 0
prompt_system = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: 23-07-01"""

# input your api_key
openai.api_key = sys.argv[1]
# input your task name to select the input file
name_task = sys.argv[2]
# each question should have integrated Input, and unique ID


def get_response(input):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        # max_tokens = 1,
        messages=[
            # {"role": "system", "content": prompt_system},
            {"role": "user", "content": input},
        ],
        timeout=60,
    )
    return response


def get_result_prediction(dict_test):
    input = dict_test["Input"]
    id = dict_test["ID"]
    if not dict_test.get("usage", None):
        try:
            response = get_response(input)
            # result and prediction
            result = response["choices"][0]["message"]["content"]
            dict_test["Result"] = result
            # usage
            dict_test["usage"] = response["usage"]._previous
            # finish_reason
            dict_test["finish_reason"] = response["choices"][0]["finish_reason"]
        except:
            print(f"Wrong: {id}\n")
    else:
        print(f"Skip: {id}\n")

    return dict_test


def cal_cost(token_input, token_output, cost_input_1k, cost_output_1k):
    cost_input = token_input * cost_input_1k / 1000 * 7.24
    cost_output = token_output * cost_output_1k / 1000 * 7.24
    print(f"cost_input: {cost_input:.2f} rmb\n")
    print(f"cost_output: {cost_output:.2f} rmb\n")
    print(f"All cost: {(cost_input+cost_output):.2f} rmb\n")


def main():
    max_thread = 5
    with open(f"../result/{name_task}.json", "r", encoding="utf-8") as f:
        list_dict_test = json.load(f)
    data = list_dict_test
    print(f"ready for {len(data)} data\n")
    # data = list_dict_test
    num_threads = len(data) // 2
    num_threads = max_thread if num_threads > max_thread else num_threads
    # time cost
    time_start = time.time()
    pool = multiprocessing.Pool(num_threads)
    list_dict_test_new = pool.map(get_result_prediction, tqdm(data))
    pool.close()
    pool.join()
    time_cost = time.time() - time_start
    print(
        f"messages length:{len(data)}, num_threads:{num_threads}, time cost:{time_cost}\n"
    )
    # save first
    with open(f"../result/{name_task}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(list_dict_test_new, indent=2, ensure_ascii=False))

    # supplement failed
    data_failed = [
        dict_test for dict_test in list_dict_test_new if not dict_test.get("Result")
    ]
    print(f"Supplement {len(data_failed)} failed\n")
    if len(data_failed) > max_thread:
        num_threads = len(data_failed) // 2
        num_threads = max_thread if num_threads > max_thread else num_threads
        pool = multiprocessing.Pool(num_threads)
        list_dict_test_supp = pool.map(get_result_prediction, tqdm(data_failed))
        pool.close()
        pool.join()
        for dict_supp in list_dict_test_supp:
            for idx_test, dict_test in enumerate(list_dict_test_new):
                if dict_test["ID"] == dict_supp["ID"]:
                    break
            list_dict_test_new[idx_test] = dict_supp
    else:
        for dict_supp in tqdm(data_failed):
            for idx_test, dict_test in enumerate(list_dict_test_new):
                if dict_test["ID"] == dict_supp["ID"]:
                    break
            list_dict_test_new[idx_test] = get_result_prediction(dict_supp)

    # count of get response
    data_fined = [
        dict_test for dict_test in list_dict_test_new if dict_test.get("Result")
    ]
    print(f"Get fined response:{len(data_fined)}\n")

    # save final
    with open(f"../result/{name_task}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(list_dict_test_new, indent=2, ensure_ascii=False))

    # cost
    token_input = sum([dict_test["usage"]["prompt_tokens"] for dict_test in data_fined])
    token_output = sum(
        [dict_test["usage"]["completion_tokens"] for dict_test in data_fined]
    )
    cal_cost(token_input, token_output, cost_input_1k, cost_output_1k)


if __name__ == "__main__":
    print(f"name_task:{name_task}\n")
    main()
