from collections import OrderedDict, Counter
import networkx as nx
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
import string
punctuations = string.punctuation
from SSRank.utils import get_noun_chunks


def compute_score(data, upper_alpha=3, title_alpha=2, window_size=3, damping=0.9):
    text = data["text"]
    candidate_phrases = get_noun_chunks(text)

    candidate_words = []
    for phrase in candidate_phrases:
        for lemma in phrase.split():
            if lemma not in punctuations:
                candidate_words.append(lemma)


    """Build token_pairs from windows in sentences"""
    token_pairs = list()
    for sent in nlp(text).sents:
        for i, token in enumerate(sent):
            for j in range(i + 1, i + window_size):
                if j >= len(sent):
                    break
                if token.lemma_ in candidate_words and sent[j].lemma_ in candidate_words:
                    pair = (token.lemma_, sent[j].lemma_)
                    if pair[0] != pair[1]:
                        token_pairs.append(pair)

    c = Counter(token_pairs)
    adj_list = []
    for key, value in enumerate(c):
        adj_tuple = value[0], value[1], c[value]
        adj_list.append(adj_tuple)


    df = pd.DataFrame(adj_list, columns=["source", "target", "weight"])
    G_weighted = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph, edge_attr='weight')
    G_weighted.add_nodes_from(candidate_words)
    weighted_pagerank = nx.pagerank(G_weighted, alpha=damping)

    title = data["title"]
    new_weighted_pagerank = {}
    for key in weighted_pagerank:
        if key in title:
            new_weighted_pagerank[key] = weighted_pagerank[key] * title_alpha
        if key.isupper:
            new_weighted_pagerank[key] = weighted_pagerank[key] * upper_alpha
        else:
            new_weighted_pagerank[key] = weighted_pagerank[key]

    phrase_score_dict = {}
    for phrase in candidate_phrases:
        score = 0
        for word in phrase.split():
            try:
                new = new_weighted_pagerank[word]
                score += new
            except:
                continue
        phrase_score_dict[phrase] = score

    new_phrase_score_dict = {}
    phrases = []

    for phrase1, score1 in phrase_score_dict.items():
        for phrase2, score2 in phrase_score_dict.items():
            if phrase1 in phrase2 and phrase1 != phrase2:
                if phrase1 not in new_phrase_score_dict:
                    new_phrase_score_dict[phrase1] = score1
                if phrase2 not in new_phrase_score_dict:
                    new_phrase_score_dict[phrase2] = score2
                if score1 > score2:
                    new_phrase_score_dict[phrase1] += score2
                    del new_phrase_score_dict[phrase2]
                    phrases.append(phrase2)
                else:
                    new_phrase_score_dict[phrase2] += score1
                    del new_phrase_score_dict[phrase1]
                    phrases.append(phrase1)

    total = {}
    for key in phrase_score_dict:
        if key not in phrases:
            if key in new_phrase_score_dict:
                total[key] = new_phrase_score_dict[key]
            else:
                total[key] = phrase_score_dict[key]


    total = OrderedDict(sorted(total.items(), key=lambda t: t[1], reverse=True))
    return total




if __name__ == '__main__':
    from SSRank.utils import data_loader
    data = data_loader(5, r'D:\tyg_research\code 2.0\SSRank\data\Inspec')
    print(compute_score(data))
    print(data["present"])
    print(data["text"])
