import time
import argparse

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from flask import Flask, request

from lm_utils import concat, get_sentences, ConfusionSet, format_info


class LMCorrector_(object):
    def __init__(self, lm_name, alpha):
        self.lm_name = lm_name
        self.alpha = alpha

        device = "cuda:0"
        self.scorer = LMScorer.from_pretrained(
           lm_name, device=device, batch_size=1
        )
        self.confusion_set = ConfusionSet()
        self.color_to_num = {
            "blue": "#8ef",
            "red": "#faa",
            "green": "#afa",
            "yellow": "#8ef",
        }


    def __call__(self, text):
        begin = time.time()
        output, words_annotated, confusion_info = self._correct(text)
        res = {}
        res["time_spent"] = time.time() - begin
        res["words_annotated"] = words_annotated
        res["output"] = output
        # res["confusion_info"] = format_info(confusion_info)
        prefix = '<div class="textContainer_Truncate">'
        suffix = "</div>"
        res["confusion_info"] = prefix + " ".join(
            f'<p class="card-text">{t}</p>' for t in confusion_info) + suffix
        return res


    def _correct(self, text):
        words = text.strip().split()
        words_ori = text.strip().split()
        prev = self.scorer.sentence_score(text, log=True)
        words_annotated = []
        confusion_info = ["<strong>涉及的混淆集：</strong>"]
        for ind, token in enumerate(words):
            clist = list(self.confusion_set.get(token.lower()))
            if not clist:
                words_annotated.append((token, "", self.color_to_num["green"]))
                continue
            
            sentences, clist = get_sentences(words, ind, clist)
            cinfo = f"<strong>{token}: </strong> " + "/".join(clist)
            confusion_info.append(cinfo)
            scores = self.scorer.sentence_score(sentences, log=True)
            assert len(scores) == len(clist)
            # best score
            best_ind, best_lp = 0, float('-inf')
            for i, lp in enumerate(scores):
                if lp > best_lp:
                    best_ind = i
                    best_lp = lp

            if (best_lp - prev) > self.alpha:
                words[ind] = clist[best_ind]
                prev = best_lp
                replacement = clist[best_ind] if clist[best_ind] else "$D$"
                words_annotated.append(
                    (words_ori[ind], replacement, self.color_to_num["red"]))
            else:
                words_annotated.append(
                    (words_ori[ind], "", self.color_to_num["green"]))
        return concat(words), words_annotated, confusion_info


app = Flask(__name__)
# model-name  ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", distilgpt2"]


@app.route('/api/lm', methods=['POST'])
def correct_by_lm():
    text = request.form["text"]
    res = lm_corrector(text)
    return res


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_name',
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"],
                        help='Name of the language model.')
    parser.add_argument('--alpha', type=int, help='替换阈值', default=2)
    parser.add_argument('--port', type=int, help='运行端口', default=8848)
    args = parser.parse_args()
    lm_corrector = LMCorrector_(args.lm_name, args.alpha)
    app.run(port=args.port)
