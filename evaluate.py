from SSRank.utils import data_loader, get_noun_chunks
from SSRank.select import get_keywords
from SSRank.clustering import HAC_clustering, K_means
import os
import multiprocessing
from functools import partial
import re
from nltk.stem import PorterStemmer

ps = PorterStemmer()
import copy


def save(i, dataset_dir, variable, value):
    data = data_loader(i, dataset_dir)
    try:
        kwargs = {variable: value}
        predictions = get_keywords(data, **kwargs)["keywords"].keys()
    except:
        predictions = get_noun_chunks(data["text"])
    return '\n'.join(predictions)


def stemmed(phrases_list):
    stemmed_list = []
    for phrase in phrases_list:
        new_phrase = ' '.join(ps.stem(word) for word in re.split(r"[ \-\t]+", phrase) if word)
        stemmed_list.append(new_phrase)
    return stemmed_list


def partial_matching(pred, gold):
    common = []
    for i in pred:
        if i in gold:
            common.append(i)
            gold.remove(i)
        else:
            for j in gold:
                if (i in j or j in i) and i not in common:
                    common.append(i)
                    gold.remove(j)
    return common


def exact_matching(lst1, lst2):
    return [value for value in lst1 if value in lst2]


def merge_lists(l1):
    l2 = []
    for i in l1:
        if i:
            for j in i:
                l2.append(j)
    return l2


def keys_loader(dataset_dir, dir_name, num):
    keys_list = []
    filename_list = os.listdir(os.path.join(dataset_dir, 'docsutf8'))
    for i in range(num):
        with open(os.path.join(dataset_dir, dir_name, filename_list[i][:-4] + '.key'), encoding="utf-8") as f:
            keys = f.read().split("\n")
        keys_list.append(stemmed(keys))
    return keys_list


def compute_metrics(keys_list, predictions_list, matching):
    common_list = []

    for i in range(len(predictions_list)):
        keys_list_copy = copy.deepcopy(keys_list)
        common_list.append(matching(predictions_list[i], keys_list_copy[i]))
    new_common_list = merge_lists(common_list)
    new_keys_list = merge_lists(keys_list)
    new_predictions_list = merge_lists(predictions_list)

    precision = len(new_common_list) / len(new_predictions_list)
    recall = len(new_common_list) / len(new_keys_list)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_all_metrics(datasets, variables, num, match_method, data_dir, save_result=True):
    metrics_dict = {}
    mapping = {'partial_matching': partial_matching, 'exact_matching': exact_matching, "HAC_clustering": HAC_clustering,
               "K_means": K_means}
    for dataset in datasets:
        dataset_dir = f"{data_dir}/{dataset}"
        keys_list = keys_loader(dataset_dir, "keys", num)
        for variable in variables:
            values = variables[variable]
            for value in values:
                if save_result:
                    new_dir = f"{dataset_dir}/{variable}={value}"
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    if variable == "clustering":
                        new_value = mapping[value]
                        new_save = partial(save, dataset_dir=dataset_dir, variable=variable, value=new_value)
                    else:
                        new_save = partial(save, dataset_dir=dataset_dir, variable=variable, value=value)
                    pool = multiprocessing.Pool()
                    results = pool.map(new_save, range(num))
                    pool.close()
                    for j in range(num):
                        with open(f"{new_dir}/{data_loader(j, dataset_dir)['file']}.key", "w", encoding="utf-8") as f:
                            f.write(results[j])
                    print(f"Results saved for {dataset} with {variable}={value}!")

                dir_name = f"{variable}={value}"
                predictions_list = keys_loader(dataset_dir, dir_name, num)
                metrics = compute_metrics(keys_list, predictions_list, matching=mapping[match_method])
                print(
                    f"The {match_method} precision, recall and f1 score for {dataset} with {variable}={value} is {metrics}")
                metrics_dict[(dataset, variable, value)] = metrics
    return metrics_dict


# Testing cases
if __name__ == '__main__':
    datasets = ['Inspec', 'SemEval2017', 'KDD', 'WWW', 'JML']
    variables = {
        'window_size': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'damping': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        'clustering': ["HAC_clustering", "K_means"]}
    num = 400
    match_method = 'exact_matching'
    data_dir = f"D:/tyg_research/code 2.0/SSRank/data"
    metrics_dict = compute_all_metrics(datasets, variables, num, match_method, data_dir, save_result=True)
    import pandas as pd

    df = pd.DataFrame(metrics_dict)
    df.to_csv("all_metrics.csv")
