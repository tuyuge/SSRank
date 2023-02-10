import re
from nltk.stem import PorterStemmer
from get_candidates import get_all_nps
import numpy as np
import os

ps = PorterStemmer()


def data_loader(index, dataset_name):
    dataset_dir = 'data/{}'.format(dataset_name)
    filename_list = os.listdir(os.path.join(dataset_dir, 'docsutf8'))

    with open(os.path.join(dataset_dir, 'docsutf8', filename_list[index]), encoding="utf-8") as f:
        text = f.read()
    with open(os.path.join(dataset_dir, 'keys', filename_list[index][:-4] + '.key'), encoding="utf-8") as f:
        keys = f.read()
    data = {'text': text, 'keys': keys}
    return data


def get_stemmed_keys(keys, text):
    stemmed = []
    stemmed_text = ' '.join(ps.stem(word) for word in re.split(r"[ \-\n\t]+", text) if word)
    for key in keys:
        new_key = ' '.join(ps.stem(word) for word in re.split(r"[ \-\n\t]+", key) if word)
        if new_key and new_key in stemmed_text and new_key not in stemmed:
            stemmed.append(new_key)
    return stemmed


def get_stemmed_predictions(predictions):
    stemmed_predictions = []
    for prediction in predictions:
        new_prediction = ' '.join(ps.stem(word) for word in re.split(r" |-", prediction) if word)
        if new_prediction:
            stemmed_predictions.append(new_prediction)
    return stemmed_predictions


def partial_matching(pred, gold):
    common = []
    for i in pred:
        if i in gold:
            common.append(i)
        else:
            for j in gold:
                if (i in j or j in i) and i not in common:
                    common.append(i)
    return common


def exact_matching(lst1, lst2):
    return [value for value in lst1 if value in lst2]


class Evaluate:
    def __init__(self, dataset_name="Inspec", partial_match=True, save=True):
        self.dataset_name = dataset_name
        self.partial_match = partial_match
        self.save = save

    def compute_metrics(self, i):
        # compute result of a single doc
        data = data_loader(i, self.dataset_name)
        text = data["text"]
        keys = data['keys'].split("\n")
        predictions = get_all_nps(text)
        stemmed_keys = get_stemmed_keys(keys, text)
        stemmed_predictions = get_stemmed_predictions(predictions)

        if self.partial_match:
            common_keys = partial_matching(stemmed_keys, stemmed_predictions)
        else:
            common_keys = exact_matching(stemmed_keys, stemmed_predictions)

        missed = [i for i in stemmed_keys if i not in common_keys]
        result = f"Index:{str(i)}\nKeywords missed:\n{str(missed)}\nKeywords predicted:\n{str(common_keys)}\nOriginal text:\n{repr(text)}\nYour predictions:\n{predictions}\n"

        if len(stemmed_keys) == 0 or stemmed_predictions == 0:
            return {"index": i, "result": result, "metrics": (1, 1, 1)}
        precision = len(common_keys) / len(stemmed_predictions)
        recall = len(common_keys) / len(stemmed_keys)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return {"result": result, "metrics": (precision, recall, f1)}

    def evaluate(self, num_docs):
        """Evaluate the result across a dataset"""
        result_list = []
        import concurrent.futures
        if self.save:
            f = open(f"result/{self.dataset_name}_{str(num_docs)}_result.txt", "w")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for item in executor.map(self.compute_metrics, range(num_docs)):
                    f.write(item["result"])
                    result_list.append(item["metrics"])
            f.close()
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for item in executor.map(self.compute_metrics, range(num_docs)):
                    result_list.append(item["metrics"])

        mean_precision = np.mean([i[0] for i in result_list])
        mean_recall = np.mean([i[1] for i in result_list])
        mean_f1 = np.mean([i[2] for i in result_list])
        with open(f"result/{self.dataset_name}_{str(num_docs)}_metrics.txt", "w") as f:
            f.write(
                f"Result of {self.dataset_name} at {str(num_docs)} docs:\n{mean_precision}, {mean_recall}, {mean_f1}")


# Testing
if __name__ == '__main__':
    evaluate_case = Evaluate(dataset_name="Inspec", partial_match=True, save=True)
    evaluate_case.evaluate(50)
