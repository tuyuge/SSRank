import spacy

nlp = spacy.load('en_core_web_sm')


def extract_np(psent):
    for subtree in psent.subtrees():
        if subtree.label() == 'NP':
            yield ' '.join(word for word, tag in subtree.leaves() if tag not in ['DT'])


def get_all_nps(input_text):
    doc = nlp(input_text)
    np_list = []
    grammar = r"""
        NP: {<JJ|NN.*><HYPH><JJ>*<VBG|VBN>?<NN.*|VBG>+} # with hyphen
        {<DT><RB>?<JJ>?<VBG|VBN><NN.*>+} # verbal complement 
        {<JJ>+<NN.*>+<IN|TO><NN.*>+} # like 'classical component of mind' 'language of thought'   
        {<DT><JJ|HYPH|NN.*|VBG|VBN>+<NN.*>}    # the longest sequence that starts with DT and ends with NN
        {<JJ>*<VBG|VBN>?<NN.*>+}   # the longest sequence of nouns and adjectives            
      """
    import nltk
    noun_parser = nltk.RegexpParser(grammar)
    for sent in doc.sents:
        tagged_sent = [(token.text, token.tag_) for token in sent if token.tag_ not in ['_SP']]
        # remove _sp tag for later formation of grammar regex
        parsed_sent = noun_parser.parse(tagged_sent)
        for np in extract_np(parsed_sent):
            # remove stopwords
            if ("p." not in np) and (np not in np_list):
                np_list.append(np)
    return np_list


# Testing cases
if __name__ == '__main__':
    from evaluate import data_loader

    text = data_loader(11, 'Inspec')["text"]
    print(get_all_nps(text))
