import os
import re
import spacy
nlp = spacy.load('en_core_web_sm')
def data_loader(index, dataset_dir):
    '''
    :param index: int
    :param dataset_dir: absolute path
    :return:
    '''
    filename_list = os.listdir(os.path.join(dataset_dir, 'docsutf8'))

    with open(os.path.join(dataset_dir, 'docsutf8', filename_list[index]), encoding="utf-8") as f:
        text = f.read().replace("\n", " ").replace("\t", "")
        text = re.sub(r' \(.*\)', '', text)
        title = text.split('.', maxsplit=1)[0]

    with open(os.path.join(dataset_dir, 'keys', filename_list[index][:-4] + '.key'), encoding="utf-8") as f:
        keys = [i.replace("\t", "") for i in f.read().split("\n") if i]
        present = [key for key in keys if key.lower() in text.lower()]


    data = {'text':text, 'title': title, 'keys': keys, 'present':present, 'file': filename_list[index][:-4]}
    return data

def get_noun_chunks(text):
    doc = nlp(text)
    chunks = []
    for chunk in doc.noun_chunks:
        new_chunk = ""
        for token in chunk:
            if token.pos_ not in ['DET', 'PRON'] and token.lemma_ not in get_stopwords():
                new_chunk+=token.lemma_+" "
        new_chunk =new_chunk.strip('"').strip()
        if new_chunk and new_chunk not in chunks:
            chunks.append(new_chunk)
    return chunks


def get_stopwords():
    from nltk.corpus import stopwords
    add = {"one", "two", "three", "four", "five", "even", "well", "good"}
    return set(stopwords.words('english')).union(add)


# Testing cases
if __name__ == '__main__':
    data = data_loader(18, r'D:\tyg_research\code 2.0\SSRank\data\JML')
    print(data['title'])
    print(data['text'])
    print(get_noun_chunks(data['text']))
    print(data["present"])