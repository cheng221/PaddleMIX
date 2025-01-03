"""Response Parsing and Evaluation for various models"""
import random
import re
from typing import Dict
random.seed(42)
import numpy as np


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = ' ' + response + ' '
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f' {can} ')
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]
    return pred_index


def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    string = string.strip()
    is_number = check_is_number(string)
    if is_number:
        string = string.replace(',', '')
        string = float(string)
        string = round(string, 2)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [' ' + string, string + ' ']
        return [string]


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    pattern_commas = '-?\\b\\d{1,3}(?:,\\d{3})+\\b'
    pattern_scientific = '-?\\d+(?:\\.\\d+)?[eE][+-]?\\d+'
    pattern_simple = (
        '-?(?:\\d+\\.\\d+|\\.\\d+|\\d+\\b)(?![eE][+-]?\\d+)(?![,\\d])')
    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip('.').lower()
        sub_responses = re.split('\\.\\s(?=[A-Z])|\\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ', 'thus ',
            'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1
                            ].strip()
                    elif len(resp.split(indicator)[-1].strip()) < len(
                        shortest_key_response):
                        shortest_key_response = resp.split(indicator)[-1
                            ].strip()
            if shortest_key_response:
                if shortest_key_response.strip() not in [':', ',', '.', '!',
                    '?', ';', ':', "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses
    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list
    pred_list = list(set(pred_list))
    return pred_list


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    elif gold_i == pred_i:
        correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        elif pred in norm_answers:
            if not correct:
                correct = True
            break
    return correct


def evaluate(samples):
    """
    Batch evaluation for multiple choice and open questions.
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample['answer']
        pred_i = sample['parsed_pred']
        if sample['question_type'] == 'multiple-choice':
            correct = eval_multi_choice(gold_i, pred_i)
        else:
            correct = eval_open(gold_i, pred_i)
        if correct:
            judge_dict[sample['id']] = 'Correct'
            pred_correct += 1
        else:
            judge_dict[sample['id']] = 'Wrong'
    if len(samples) == 0:
        return {'acc': 0}
    return judge_dict, {'acc': pred_correct / len(samples)}


def calculate_ins_level_acc(results: Dict):
    """Calculate the instruction level accuracy for given Subject results"""
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results['acc'] * cat_results['num_example']
        ins_num += cat_results['num_example']
    if ins_num == 0:
        return 0
    return acc / ins_num
