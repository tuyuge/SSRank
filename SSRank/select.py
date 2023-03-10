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

    data = data_loader(18, r'D:\tyg_research\code 2.0\SSRank\data\JML')
    ssrank = get_keywords(data, upper_alpha=3, title_alpha=2, window_size=3, damping=0.9, clustering=HAC_clustering, n=2)

    for keyword in ssrank["keywords"]:
        print(keyword)
    print(ssrank["candidates"])
    print(ssrank["rank"])
    print(ssrank["clusters"])
    print(ssrank["keywords"])
    print(data["present"])
    print(data["text"])
