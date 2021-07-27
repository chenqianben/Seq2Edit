from os.path import join 

from tqdm import tqdm
import spacy

from corrector import MultiCorrector
from config import Config

NLP = spacy.load('/home/LAB/luopx/bea2019/confusion_set/lm_score/en_core_web_sm-2.3.1/en_core_web_sm/en_core_web_sm-2.3.1/')

def has_suitable_length(line, min_word_cnt, max_word_cnt):
    return min_word_cnt <= len(line.strip().split()) <= max_word_cnt


def get_type(input_text, rule_out, lm_out, s2e_out):
    """
    # 三个都有修正: all
    # 规则修正了，模型没有修正: rule
    # 规则没有修正 模型修正了: model
    """
    input_text = normalize(input_text)
    rule_out = normalize(rule_out)
    lm_out = normalize(lm_out)
    s2e_out = normalize(s2e_out)
    if rule_out != input_text and lm_out != rule_out and s2e_out != lm_out:
        return "all"
    elif  rule_out != input_text and lm_out == rule_out and s2e_out == rule_out:
        return "rule"
    elif  rule_out == input_text and ((lm_out != rule_out) or (s2e_out != rule_out)):
        return "model"
    return "none"


def get_examples_from_file(data_dir, split):
    corrector = MultiCorrector(Config)
    src_file = join(data_dir, f"{split}.src")
    tgt_file = join(data_dir, f"{split}.trg")

    examples = {
        "all": [],
        "rule": [],
        "model": [],
    }
    with open(src_file) as src, open(tgt_file) as tgt:
        for s, t in zip(src, tgt):
            if has_suitable_length(s, 6, 15):
                text = s.strip()
                outputs = corrector(text, return_outputs=True)
                outputs.insert(0, text)
                cor_type = get_type(*outputs)
                if cor_type in examples:
                    examples[cor_type].append(text)

    # 写入
    with open("examples.txt", "w") as f:
        for k in examples:
            for sent in examples[k]:
                f.write(f"{k}:: {sent}\n")



def normalize(text):
    doc = NLP(text)
    tokens = []
    for token in doc:
        if token.text:
            tokens.append(token.text)
    return " ".join(tokens)



if __name__ == "__main__":
    data_dir = "/home/LAB/luopx/bea2019/mlconvgec2018/training/processed_word"
    split = "test"
    get_examples_from_file(data_dir, split)
