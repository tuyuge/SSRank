import numpy as np
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
from collections import OrderedDict

from sklearn.cluster import KMeans

def editDistDP(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j

            elif j == 0:
                dp[i][j] = i

            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                   dp[i - 1][j],
                                   dp[i - 1][j - 1])

    return 1 - dp[m][n] / max(m, n)


def HAC_clustering(candidate_phrases, n=2):
    '''

    :param text:
    :param n: n_clusters=int(size/n)
    :return:
    '''

    size = len(candidate_phrases)

    distance_matrix = np.zeros((size, size), dtype='float')
    for i in range(size):
        for j in range(size):
            distance_matrix[i][j] = editDistDP(candidate_phrases[i], candidate_phrases[j])

    hiercluster = AgglomerativeClustering(affinity='euclidean', n_clusters=int(size/n), linkage='ward', compute_full_tree=True)
    # Fit the data to the model and determine which clusters each data point belongs to:
    model = hiercluster.fit(distance_matrix)
    clusters = model.labels_

    candidates_clusters = {}
    for i in range(int(size/2)):
        index = np.where(clusters == i)
        import itertools
        score = 0
        for t in itertools.combinations(index[0], 2):
            score += distance_matrix[t[0]][t[1]]
        score = score/len(index[0])


        candidates_clusters['| '.join(candidate_phrases[i] for i in index[0])] = score
    sorted_candidates_clusters = OrderedDict(sorted(candidates_clusters.items(), key=lambda t: t[1], reverse=True))
    return sorted_candidates_clusters


def phrase_embed(p, model):
    word_embed_list = []
    for i in p.split():
        try:
            word_embed_list.append(model[i])
        except:
            continue
    return np.mean(word_embed_list, axis=0)

def K_means(candidate_phrases, n=2):
    '''

    :param text:
    :param n: n_clusters=int(size/n)
    :return:
    '''
    import gensim.downloader as api
    model = api.load("glove-wiki-gigaword-300")
    size = len(candidate_phrases)
    kmeans = KMeans(n_clusters=int(size/n))

    labels = []
    tokens = []
    for phrase in candidate_phrases:
        tokens.append(phrase_embed(phrase, model))
        labels.append(phrase)


    kmeans.fit(tokens)
    y_kmeans = kmeans.predict(tokens)
    text_mean = np.mean(tokens, axis=0)
    candidates_clusters = {}
    for i in range(int(size/n)):
        cluster = []
        cluster_embed = []
        for j in range(len(y_kmeans)):
            if y_kmeans[j] == i:
                cluster.append(labels[j])
                cluster_embed.append(tokens[j])
        cluster_mean = np.mean(cluster_embed, axis=0)

        cosine = np.dot(text_mean, cluster_mean) / (norm(text_mean) * norm(cluster_mean))
        candidates_clusters['| '.join(cluster)] = cosine

    sorted_candidates_clusters = OrderedDict(sorted(candidates_clusters.items(), key=lambda t: t[1], reverse=True))
    return sorted_candidates_clusters


if __name__ == '__main__':
    from SSRank.utils import data_loader
    from SSRank.rank import compute_score
    data = data_loader(5, r'D:\tyg_research\code 2.0\SSRank\data\Inspec')
    sorted_candidates_scores = compute_score(data)
    candidate_phrases = list(sorted_candidates_scores.keys())
    print(K_means(candidate_phrases))
    print(data["present"])