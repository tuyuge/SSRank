from get_candidates import get_nps
import os
import multiprocessing


class Save:
    def __init__(self, dataset_name):
        self.datasetname = dataset_name
        self.dataset_dir = f'data/{dataset_name}'
        self.new_dir = f"{self.dataset_dir}/np_results"
        self.all_files = os.listdir(f"{self.dataset_dir}/docsutf8")


    def save(self, i):
        file = self.all_files[i]
        with open(f"{self.dataset_dir}/docsutf8/{file}", encoding="utf-8") as f:
            text = f.read()
        candidates = get_nps(text)[0]
        with open(f"{self.new_dir}/{file}", "w", encoding="utf-8") as f:
            f.write("\n".join(candidates))


    def save_all_results(self):
        if not os.path.exists(self.new_dir):
            os.makedirs(self.new_dir)
        pool = multiprocessing.Pool()
        pool.map(self.save, range(len(self.all_files)))
        pool.close()


# Testing cases
if __name__ == '__main__':
    Save("kdd").save_all_results()