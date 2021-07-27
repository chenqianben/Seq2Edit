import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import random
import time

from flask import Flask, render_template, request

from annotated_text_v2 import annotated_text
from corrector import RulesCorrector, LMCorrector, Seq2EditsCorrector
from config import Config

app = Flask(__name__)
rule_corrector = RulesCorrector(rule_file=Config.rule_file)
lm_corrector = LMCorrector(lm_name=Config.lm_name, alpha=Config.lm_alpha)
seq2edits_corrector = Seq2EditsCorrector(Config.seq2edits_encoder)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/rule', methods=['POST'])
def correct_by_rules():
    text = request.form["text"]
    res = rule_corrector(text)
    return res


# @app.route('/api/correct', methods=['POST'])
# def correct():
#     text = request.form["text"]
#     res = corrector(text)
#     return res

@app.route('/api/lm', methods=['POST'])
def correct_by_lm():
    text = request.form["text"]
    res = lm_corrector(text)
    return res


@app.route('/api/seq2edits', methods=['POST'])
def correct_by_seq2edits():
    text = request.form["text"]
    res = seq2edits_corrector(text)
    return res


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', port=12321, debug=False)
