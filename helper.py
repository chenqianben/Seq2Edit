import spacy

# ----------------------RulesCorrector helper functions-----------------------

def find_all(pa, text):
    start_points = []
    pos = text.find(pa, 0)
    while pos != -1:
        start_points.append(pos)
        pos = text.find(pa, pos+len(pa))
    return start_points

NLP = spacy.load('/home/LAB/luopx/bea2019/confusion_set/lm_score/en_core_web_sm-2.3.1/en_core_web_sm/en_core_web_sm-2.3.1/')

def tokenize(text):
    doc = NLP(text)
    tokens = []
    for token in doc:
        if token.text:
            tokens.append(token.text)
    return " ".join(tokens)
