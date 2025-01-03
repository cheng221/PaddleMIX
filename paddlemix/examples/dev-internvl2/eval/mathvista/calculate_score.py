import argparse
import sys
import pandas as pd
from Levenshtein import distance
from utilities import *


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


def normalize_extracted_answer(extraction, choices, question_type,
    answer_type, precision):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ''
        letter = re.findall('\\(([a-zA-Z])\\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()
        options = [chr(ord('A') + i) for i in range(len(choices))]
        if extraction in options:
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices
    elif answer_type == 'integer':
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None
    elif answer_type == 'float':
        try:
            extraction = str(round(float(extraction), int(precision)))
        except:
            extraction = None
    elif answer_type == 'list':
        try:
            extraction = str(extraction)
        except:
            extraction = None
    return extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def get_acc_with_contion(res_pd, key, value):
    if key == 'skills':
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = '{:.2f}'.format(len(correct_pd) / len(total_pd) * 100)
    return len(correct_pd), len(total_pd), acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--score_file', type=str, default='scores.json')
    parser.add_argument('--gt_file', type=str, default=
        './data/MathVista/annot_testmini.json', help='ground truth file')
    parser.add_argument('--number', type=int, default=-1, help=
        'number of problems to run')
    parser.add_argument('--rerun', action='store_true', help=
        'rerun the evaluation')
    parser.add_argument('--caculate_gain', action='store_true', help=
        'caculate the score gains over random guess')
    parser.add_argument('--random_file', type=str, default=
        'score_random_guess.json')
    args = parser.parse_args()
    output_file = os.path.join(args.output_dir, args.output_file)
    print(f'Reading {output_file}...')
    results = read_json(output_file)
    print(f'Reading {args.gt_file}...')
    gts = read_json(args.gt_file)
    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print('Number of testing problems:', len(full_pids))
    print('\nEvaluating the predictions...')
    update_json_flag = False
    for pid in full_pids:
        problem = results[pid]
        if args.rerun:
            if 'prediction' in problem:
                del problem['prediction']
            if 'true_false' in problem:
                del problem['true_false']
        choices = problem['choices']
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        precision = problem['precision']
        extraction = problem['extraction']
        if 'answer' in problem:
            answer = problem['answer']
        else:
            if pid in gts:
                answer = gts[pid]['answer']
            else:
                answer = ''
            problem['answer'] = answer
        prediction = normalize_extracted_answer(extraction, choices,
            question_type, answer_type, precision)
        true_false = safe_equal(prediction, answer)
        if 'true_false' not in problem:
            update_json_flag = True
        elif true_false != problem['true_false']:
            update_json_flag = True
        if 'prediction' not in problem:
            update_json_flag = True
        elif prediction != problem['prediction']:
            update_json_flag = True
        problem['prediction'] = prediction
        problem['true_false'] = true_false
    if update_json_flag:
        print('\n!!!Some problems are updated.!!!')
        print(f'\nSaving {output_file}...')
        save_json(results, output_file)
    total = len(full_pids)
    correct = 0
    for pid in full_pids:
        if results[pid]['true_false']:
            correct += 1
    accuracy = str(round(correct / total * 100, 2))
    print(f'\nCorrect: {correct}, Total: {total}, Accuracy: {accuracy}%')
    scores = {'average': {'accuracy': accuracy, 'correct': correct, 'total':
        total}}
    for pid in results:
        results[pid].update(results[pid].pop('metadata'))
    df = pd.DataFrame(results).T
    print(len(df))
    print('Number of test problems:', len(df))
    target_keys = ['question_type', 'answer_type', 'language', 'source',
        'category', 'task', 'context', 'grade', 'skills']
    for key in target_keys:
        print(f'\nType: [{key}]')
        if key == 'skills':
            values = []
            for i in range(len(df)):
                values += df[key][i]
            values = list(set(values))
        else:
            values = df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_contion(df, key, value)
            if total > 0:
                print(f'[{value}]: {acc}% ({correct}/{total})')
                scores[key][value] = {'accuracy': acc, 'correct': correct,
                    'total': total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item:
            float(item[1]['accuracy']), reverse=True))
    scores_file = os.path.join(args.output_dir, args.score_file)
    print(f'\nSaving {scores_file}...')
    save_json(scores, scores_file)
    print('\nDone!')
    if args.caculate_gain:
        random_file = os.path.join(args.output_dir, args.random_file)
        random_scores = json.load(open(random_file))
        print('\nCalculating the score gains...')
        for key in scores:
            if key == 'average':
                gain = round(float(scores[key]['accuracy']) - float(
                    random_scores[key]['accuracy']), 2)
                scores[key]['acc_gain'] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(float(scores[key][sub_key]['accuracy']) -
                        float(random_scores[key][sub_key]['accuracy']), 2)
                    scores[key][sub_key]['acc_gain'] = str(gain)
        print(f'\nSaving {scores_file}...')
        save_json(scores, scores_file)
        print('\nDone!')
