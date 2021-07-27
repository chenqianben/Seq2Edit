
from hunspell import Hunspell
from lm_scorer.models.auto import AutoLMScorer as LMScorer

# ----------------------RulesCorrector helper functions-----------------------

def find_all(pa, text):
    start_points = []
    pos = text.find(pa, 0)
    while pos != -1:
        start_points.append(pos)
        pos = text.find(pa, pos+len(pa))
    return start_points


# ----------------------LMCorrector helper functions--------------------------


class ConfusionSet(object):
    def __init__(self):
        self.preps = set([
            "about", "at", "by", "for", "from",
            "in", "of", "on", "to", "with", ""
        ])
        self.articles = set(["", "a", "an", "the"])
        self.puncts = set(list(";'.[](){}\"?!"))
        # self.hunspell = Hunspell()
        self.agid_data, self.word_to_lemma = load_agid_corpus()

    def get(self, word):
        if word in self.puncts:
            return set()

        if word in self.preps:
            return self.preps
        if word in self.articles:
            return self.articles

        confusion_set = set()
        lemma = self.word_to_lemma.get(word, word)
        confusion_set = self.agid_data.get(lemma, {})

        # 过滤非词
        return confusion_set


def load_agid_corpus():
    path = "/home/LAB/luopx/bea2019/confusion_set/pyInflect/pyinflect/infl.csv"
    print(f"Loadding Agid from {path}")
    data = {}
    word_to_lemma = {}
    cnt = 0
    with open(path) as f:
        for line in f:
            words = []
            for i, item in enumerate(line.strip().split(',')):
                if item == "<>" or i == 1:
                    continue
                item = item.split('/')
                words.extend(item)

            if not words:
                continue

            words.sort()
            lemma, confusion_set = words[0], set(words)
            if lemma in data:
                data[lemma].update(confusion_set)
            else:
                data[lemma] = confusion_set

            for word in confusion_set:
                if word in word_to_lemma and word_to_lemma[word] != lemma:
                    # print(f"Inflicts: f{word} -> {word_to_lemma[word]} -> {lemma}")
                    cnt += 1
                word_to_lemma[word] = lemma

    print(f"Inflicts {cnt}")
    return data, word_to_lemma

def concat(tokens):
    return " ".join([t for t in tokens if t])


def correct(inp_line, confusion_set, alpha, scorer):
    # prev = scorer.sentence_score(inp_line, log=True, reduce="mean")
    tokens = inp_line.strip().split()
    prev = scorer.sentence_score(inp_line, log=True)
    for ind, token in enumerate(tokens):
        clist = list(confusion_set.get(token.lower()))
        if not clist:
            continue

        sentences, clist = get_sentences(tokens, ind, clist)
        scores = scorer.sentence_score(sentences, log=True)
        assert len(scores) == len(clist)
        # best score
        best_ind, best_lp = 0, float('-inf')
        for i, lp in enumerate(scores):
            if lp > best_lp:
                best_ind = i
                best_lp = lp
        if (best_lp - prev) > alpha:
            tokens[ind] = clist[best_ind]
            prev = best_lp
    return concat(tokens)

def correct_faster(inp_line, confusion_set, alpha, scorer, topk=3):
    """
    如果bpe编码后tokens长度与原来相同，那么可以进行加速
    """
    tokens = inp_line.strip().split()

    tokens_score, _, _ = scorer.tokens_score(inp_line.strip(), log=True)
    if len(tokens_score) != (len(tokens) + 1): # 有单词被bpe切分，无法加速
        return correct(inp_line, confusion_set, alpha, scorer)

    # 只选取三个位置 进行替换
    prev = sum(tokens_score)
    tokens_score = [(s, i) for i, s in enumerate(tokens_score[:-1])]

    tokens_score.sort(key=lambda x:x[0])
    positions = [ind for _, ind in tokens_score[:topk]]
    positions.sort()

    for ind in positions:
        token = tokens[ind]
        clist = list(confusion_set.get(token.lower()))
        if not clist:
            continue

        sentences, clist = get_sentences(tokens, ind, clist)
        scores = scorer.sentence_score(sentences, log=True)
        assert len(scores) == len(clist)
        # best score
        best_ind, best_lp = 0, float('-inf')
        for i, lp in enumerate(scores):
            if lp > best_lp:
                best_ind = i
                best_lp = lp
        if (best_lp - prev) > alpha:
            tokens[ind] = clist[best_ind]
            prev = best_lp
    return concat(tokens)

def get_sentences(tokens, i, clist):
    if i == 0:
        clist = [c.capitalize() for c in clist]
    sentences = [concat(tokens[:i]+[c]+tokens[i+1:]) for c in clist]
    return sentences, clist


def fine_tune_alpha(model_name="lm_gpt2", split="dev", faster=False):
    print(f"model_name ： {model_name}, split:{split} faster: {faster}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 10
    scorer = LMScorer.from_pretrained(model_name[3:], device=device, batch_size=batch_size)
    os.makedirs(f"./data/{model_name}", exist_ok=True)
    log_file = os.path.join("logs", f"{model_name}_{split}.log")
    with open(log_file, "w") as f:
        for alpha in {0, 2, 4, 6, 8}:
            p, r, m2 = cal_m2_score_by_alpha(f"./data/{split}.src", f"./data/{model_name}/{split}.{alpha}.corrected",
                                             f"./data/{split}.m2", alpha, scorer,
                                             faster=faster)
            msg = f"alpha: {alpha}, Precision: {p}, Recall: {r}, M2:{m2}"
            f.write(msg+"\n")
            # 写到结果文件中
            print(msg)


def format_info(confusion_info):
    out = ""
    for i in range(0, len(confusion_info), 2):
        s = get_two_cols(confusion_info, i)
        out += f'<p class="card-text">{s}</p>'
    return out



def get_two_cols(confusion_list, ind, col_len=70):
    size = len(confusion_list)
    if ind == (size - 1):
        return confusion_list[ind]
    
    res = confusion_list[ind]
    while len(res) < col_len:
        res += " "
    res += confusion_list[ind+1]
    return res


