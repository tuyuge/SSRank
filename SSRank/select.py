from SSRank.clustering import HAC_clustering,K_means
from SSRank.rank import compute_score
from collections import OrderedDict


def common_words(phrase_list):
    phrase_list = [phrase.lower() for phrase in phrase_list]
    words_list = [phrase.split() for phrase in phrase_list]
    unique_words_list = [set(words) for words in words_list]
    common_words = set.intersection(*unique_words_list)
    return len(common_words)


def get_keywords(data, upper_alpha=3, title_alpha=2, window_size=3, damping=0.9, clustering=HAC_clustering, n=2):
    sorted_candidates_scores = compute_score(data, upper_alpha=upper_alpha, title_alpha=title_alpha, window_size=window_size, damping=damping)
    candidate_phrases = list(sorted_candidates_scores.keys())
    sorted_candidates_clusters = clustering(candidate_phrases, n=n)

    cluster_list = []
    for cluster in sorted_candidates_clusters:
        cluster_dict = {}
        for phrase in cluster.split("| "):
            cluster_dict[phrase] = sorted_candidates_scores[phrase]
            sorted_cluster_dict = OrderedDict(sorted(cluster_dict.items(), key=lambda t: t[1], reverse=True))
        cluster_list.append(sorted_cluster_dict)


    keywords_list = []
    for i in range(len(cluster_list) - 1):
        phrase_list = list(cluster_list[i].keys())
        num_common = common_words(phrase_list)
        if num_common > 1:
            keywords_list.append(phrase_list.pop(0))
        else:
            keywords_list.append(phrase_list.pop(0))
            if phrase_list:
                if cluster_list[i][phrase_list[0]] >= list(cluster_list[i + 1].values())[0]:
                    keywords_list.append(phrase_list.pop(0))
    keywords_list.append(list(cluster_list[-1].keys())[0])
    keywords_dict = {}
    for keyword in keywords_list:
        keywords_dict[keyword] = sorted_candidates_scores[keyword]

    sorted_keywords_dict = OrderedDict(sorted(keywords_dict.items(), key=lambda t: t[1], reverse=True))

    return {"candidates":candidate_phrases, "rank":sorted_candidates_scores, "clusters":cluster_list, "keywords":sorted_keywords_dict}


if __name__ == '__main__':
    from SSRank.utils import data_loader
    import os

    dataset = "Inspec"
    dataset_dir =  f'D:/tyg_thesis/SSRank-main/data/{dataset}'
    dir_name = "clustering/K_means"
    for i in range(10):
        data = data_loader(i,dataset_dir)
        file = data["file"]
    # data = {'title': 'Token frequency as a determinant of morphological change',
    #         'text': 'This paper demonstrates that morphological change tends to involve the replacement of low frequency forms in inflectional paradigms by innovative forms based on high frequency forms, using Greek data involving the diachronic reorganisation of verbal inflection classes. A computational procedure is outlined for generating a possibility space of morphological changes which can be represented as analogical proportions, on the basis of synchronic paradigms in ancient Greek. I then show how supplementing analogical proportions with token frequency information can help to predict whether a hypothetical change actually took place in the language’s subsequent development. Because of the crucial role of inflected surface forms serving as analogical bases in this model, I argue that the results support theories in which inflected forms can be stored whole in the lexicon.'}
        ssrank = get_keywords(data, upper_alpha=3, title_alpha=2, window_size=3, damping=0.9, clustering=K_means, n=2)
        try:
            keys = ssrank["keywords"]
        except:
            print(f"Wrong file: {file}")
        with open(os.path.join(dataset_dir, dir_name, file + '.key'), "w", encoding="utf-8") as f:
            f.write("\n".join(keys))
        print(f"Saved for the {i}th doc")
    # print(data["file"])
