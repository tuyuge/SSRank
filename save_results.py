from SSRank.utils import data_loader, get_noun_chunks
from SSRank.select import get_keywords
import os
import multiprocessing
from functools import partial


def save(i, dataset_dir, variable, value):
    data = data_loader(i, dataset_dir)
    try:
        kwargs = {variable: value}
        predictions = get_keywords(data, **kwargs)["keywords"].keys()
    except:
        predictions = get_noun_chunks(data["text"])
    return '\n'.join(predictions)


# Testing cases
if __name__ == '__main__':
    datasets = ['Inspec', 'SemEval2017', 'KDD', 'WWW', 'JML']
    variables = {'window_size':[2, 3, 4, 5, 6, 7, 8, 9, 10], 'damping':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9], 'clustering':['HAC_clustering', 'K_means']}
    num = 400
    for variable in variables:
        values = variables[variable]
        for dataset in datasets:
            dataset_dir = f"D:/tyg_research/code 2.0/SSRank/data/{dataset}"
            for value in values:
                new_dir = f"{dataset_dir}/{variable}={value}"
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                new_save = partial(save, dataset_dir=dataset_dir, variable=variable, value=value)
                pool = multiprocessing.Pool()
                results = pool.map(new_save, range(num))
                pool.close()
                for j in range(num):
                    with open(f"{new_dir}/{data_loader(j, dataset_dir)['file']}.txt", "w", encoding="utf-8") as f:
                        f.write(results[j])
                print(f"Results saved for {dataset} with {variable}={value}!")
