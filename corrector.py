import os
import time
import sys
from pprint import pprint

sys.path.append("/home/LAB/luopx/bea2019/seq2edits")
sys.path.append("/home/LAB/luopx/bea2019/gector")

import language_tool_python
import requests
import spacy

from annotated_text_v2 import annotated_text
from helper import find_all, tokenize
from preprocess_v2 import get_eidts
from gector.gec_model import GecBERTModel


class Corrector(object):
    def __init__(self):
        self.color_to_num = {
            "blue": "#8ef",
            "red": "#faa",
            "green": "#afa",
            "yellow": "#8ef",
        }
        self.html_template = """
        <div class="card">
            <div class="card-header">
            {mname}模块处理完毕，用时{time_spent:.3f}秒
            </div>
            <div class="card-body">
                <table class="table">
                    <tbody>
                      <tr>
                        <th scope="row">模型输入</th>
                        <td>{input}</td>
                      </tr>
                      <tr>
                        <th scope="row">模型处理</th>
                        <td>{annotated}</td>
                      </tr>
                      <tr>
                        <th scope="row">模型输出</th>
                        <td>{output}</td>
                      </tr>
                    </tbody>
                </table>
             </div>
        </div>
        """

    def __call__(self, text):
        raise NotImplementedError


class RulesCorrector(Corrector):
    def __init__(self, rule_file=""):
        super().__init__()
        print(f"Initialize RulesCorrector...")
        self.rule_dict = {}
        if rule_file:
            with open(rule_file) as f:
                for line in f:
                    src, tgt = line.strip().split("-->")
                    src = src.strip()
                    tgt = tgt.strip()
                    self.rule_dict[src] = tgt

        self.lang_tool = language_tool_python.LanguageTool('en-US')
        self.lang_tool.check("ttt")  # 启动
        print("RulesCorrector is ready!")


    def __call__(self, text):
        """
        先使用rule_dict过一遍
        再使用lang_tool
        """
        begin = time.time()
        lt_infos = self._correct_by_lt(text)
        rule_infos = self._correct_by_rules(text)
        infos = lt_infos + rule_infos
        infos.sort(key=lambda info: info[0])

        # 返回annotated_text函数所需要的格式
        tokens = []
        cur_pos = 0
        output = ""
        for start, end, replacement in infos:
            if start < cur_pos:
                continue  # quit this correction
            if start > cur_pos:
                span = text[cur_pos:start]
                if span.strip():  # 全是空白字符就不添加了
                    tokens.append((span, "", self.color_to_num["green"]))
                output += span
            tokens.append((text[start:end], replacement, self.color_to_num["red"]))
            cur_pos = end
            output += replacement

        span = text[cur_pos:]
        if span.strip():
            tokens.append((span, "", self.color_to_num["green"]))
        output += span
        annotated = annotated_text(*tokens)

        res = {}
        res["html"] = self.html_template.format(
            mname="规则",
            time_spent=time.time()-begin,
            input=text,
            annotated=annotated,
            output=output
        )
        res["output"] = output

        return res

    def _correct_by_rules(self, text):
        infos = []
        for src in self.rule_dict:
            start_points = find_all(src, text)
            for start in start_points:
                infos.append((start, start+len(src), self.rule_dict[src]))
        return infos

    def _correct_by_lt(self, text):
        matches = self.lang_tool.check(text)
        infos = []
        for match in matches:
            start = match.offset
            end = start + match.errorLength
            if not match.replacements:
                continue
            label = f"{match.replacements[0]}"
            infos.append((start, end, label))
        return infos


class LMCorrector(Corrector):
    def __init__(self, lm_name, alpha, port=8847):
        super().__init__()
        assert lm_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]
        print(f"Initialize LMCorrector...")
        os.system(f"lsof -ti :{port} | xargs --no-run-if-empty kill -9")
        cmd = f"nohup /home/LAB/luopx/.virtualenvs/lm_score/bin/python3 lm_corrector.py"
        cmd += f" --lm_name {lm_name} --alpha {alpha} --port {port} > lm.out &"
        os.system(cmd)
        # time.sleep(5)
        self.url = f'http://localhost:{port}/api/lm'
        self._test_url()
        print("LM loaded!")

    def __call__(self, text):
        data = {'text': tokenize(text)}
        res_ = requests.post(self.url, data=data).json()
        words = [tuple(x) for x in res_["words_annotated"]]
        annotated = annotated_text(*words)

        annotated += res_["confusion_info"]
        res = {}
        res["html"] = self.html_template.format(
            mname="语言模型",
            time_spent=res_["time_spent"],
            input=text,
            annotated=annotated,
            output=res_["output"]
        )
        res["output"] = res_["output"]
        return res

    def _test_url(self):
        data = {"text": "abc"}
        time.sleep(3)
        while True:
            try:
                res_ = requests.post(self.url, data=data)
                if res_.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(3)



class Seq2EditsCorrector(Corrector):
    def __init__(self, model_name):
        super().__init__()
        print("Initialize Seq2EditsCorrector...")
        self.model_path = self._get_model_path(model_name)
        special_tokens_fix = 1 if model_name == "roberta" else 0
        vocab_path = os.path.join(
            "/home/LAB/luopx/bea2019/gector/",
            "data_seq2edits/output_vocabulary"
        )
        self.model = GecBERTModel(
            vocab_path=vocab_path,
            model_paths=[self.model_path],
            model_name=model_name,
            special_tokens_fix=special_tokens_fix,
            is_ensemble=False
        )
        print("Seq2EditsCorrector Loaded!")

    def __call__(self, text):
        begin = time.time()
        # text = tokenize(text) # 暂时关闭
        output = self._correct(text)
        src_words, tgt_words, edits = get_eidts(text, output, return_tokens=True)
        tokens = []
        for word, edit in zip(src_words, edits):
            if edit == "$KEEP":
                tokens.append((word, edit, self.color_to_num["green"]))
            elif edit == "$DELETE":
                tokens.append((word, edit, self.color_to_num["red"]))
            elif edit.startswith("$RE"):
                tokens.append((word, edit, self.color_to_num["yellow"]))
            elif edit.startswith("$APP"):
                tokens.append((word, edit, self.color_to_num["blue"]))

        res = {}
        annotated = annotated_text(*tokens)
        res["html"] = self.html_template.format(
            mname="Seq2Edits",
            time_spent=time.time()-begin,
            input=text,
            annotated=annotated,
            output=output
        )
        res["output"] = output
        return res


    def _correct(self, text):
        preds, _ = self.model.handle_batch([text.split()])
        return " ".join(preds[0])


    @staticmethod
    def _get_model_path(model_name):
        ckpt_dir = "/home/LAB/luopx/bea2019/gector/ckpts/"
        model_to_bestepoch = {
            "bert": 16,
            "roberta": 17,
            "xlnet": 14,
            "distilbert": 16
        }
        e = model_to_bestepoch[model_name]
        return os.path.join(
            ckpt_dir,
            f"seq2edits_{model_name}/model_state_epoch_{e}.th"
        )


class MultiCorrector(Corrector):
    def __init__(self, config):
        super().__init__()
        print("Config:")
        pprint(vars(config))
        self.correctors = []
        if config.rule_corrector_on:
            self.correctors.append(RulesCorrector(rule_file=config.rule_file))

        if config.lm_corrector_on:
            self.correctors.append(
                LMCorrector(lm_name=config.lm_name, alpha=config.lm_alpha))

        if config.seq2edits_corrector_on:
            self.correctors.append(
                Seq2EditsCorrector(config.seq2edits_encoder))
        

    def __call__(self, text, return_outputs=False):
        res = {}
        outputs = []
        html = ""
        for corrector in self.correctors:
            corrector_out = corrector(text)
            outputs.append(corrector_out["output"])
            html += corrector_out["html"]
            text = corrector_out["output"]
        res["html"] = html
        res["output"] = text

        if return_outputs:
            return outputs
        else:
            return res

