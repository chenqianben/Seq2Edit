
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
from corrector import RulesCorrector, LMCorrector, Seq2EditsCorrector, MultiCorrector
from config import Config
from helper import tokenize

from tqdm import tqdm

# rule_corrector = RulesCorrector(rule_file=Config.rule_file)
# lm_corrector = LMCorrector(lm_name=Config.lm_name, alpha=Config.lm_alpha)
# seq2edits_corrector = Seq2EditsCorrector(Config.seq2edits_encoder)


def read_test_data():
    src_path = "/home/LAB/luopx/DATA/fairseq_tor/gec_data/test.src"
    with open(src_path) as f:
        src_lines = [src.strip() for src in f.readlines()]
    return src_lines


def test_model_serial():
    examples = read_test_data()
    # save to file and score
    out_file = "./outputs/serial.out"
    with open(out_file, "w") as out:
        for example in tqdm(examples):
            example = rule_corrector(example)['output']
            example = lm_corrector(example)['output']
            example = seq2edits_corrector(example)['output']
            out.write(example+'\n')
    print("model serial:")
    print(score_file(out_file, "./outputs/serial.m2"))


def score_file(tgt_file, score_out_file):
    """根据目标文件以及m2文件 得到m2分数"""
    # 执行系统命令，可以获取执行系统命令的结果
    m2_file = "/home/LAB/luopx/DATA/fairseq_tor/gec_data/official-2014.combined.m2"
    print(f"cal m2 score for {tgt_file}, score_out_file:{score_out_file}...")
    os.system(f'python2 /home/LAB/luopx/DATA/fairseq_tor/m2scorer/m2scorer -v {tgt_file} {m2_file} > {score_out_file}')

    with open(score_out_file) as f:
        lines = [line.strip() for line in f.readlines()]
    p = float(lines[-3][-6:])
    r = float(lines[-2][-6:])
    m2 = float(lines[-1][-6:])
    return p, r, m2


def score_files(tgt_files, score_out_file):
    tgt_files_string = ",".join(tgt_files)
    m2_file = "/home/LAB/luopx/DATA/fairseq_tor/gec_data/official-2014.combined.m2"
    print(f"cal m2 score for {tgt_files_string}, score_out_file:{score_out_file}...")
    os.system(f'python2 /home/LAB/luopx/DATA/fairseq_tor/m2scorer/scripts/m2scorer_v2.py -v {tgt_files_string} {m2_file} > {score_out_file}')

    with open(score_out_file) as f:
        lines = [line.strip() for line in f.readlines()]
    p = float(lines[-3][-6:])
    r = float(lines[-2][-6:])
    m2 = float(lines[-1][-6:])
    return p, r, m2


def test_model_parallel():
    examples = read_test_data()
    # save to file and score

    def correct(corrector, out_file, tokenize_after=False):
        print(f"writing reuslts to {out_file}...")
        with open(out_file, "w") as out:
            for example in tqdm(examples):
                example = corrector(example)['output']
                if tokenize_after:
                    example = tokenize(example)
                out.write(example+'\n')

    rule_out_file = "./outputs/rule.out" # 这里应该再加个分词
    # correct(rule_corrector, rule_out_file, tokenize_after=true)

    lm_out_file = "./outputs/lm.out"
    # correct(lm_corrector, lm_out_file, tokenize_after=false)

    seq2edits_out_file = "./outputs/seq2edits.out" 
    # correct(seq2edits_corrector, seq2edits_out_file, tokenize_after=false)
    score_files([rule_out_file, lm_out_file, seq2edits_out_file], "./outputs/parallel.m2")

def temp():
    # rule_out_file = "./outputs/rule.out" # 这里应该再加个分词
    lm_out_file = "./outputs/lm.out"
    seq2edits_out_file = "./outputs/seq2edits.out" 
    score_files([lm_out_file, seq2edits_out_file], "./outputs/parallel.m2")


def test_rule1_lm2_seq2edits3():
    # /home/LAB/luopx/bea2019/confusion_set/lm_score/main_lm_scorer.py:test_rule1_lm2函数
    lm_out_file = "./outputs/lm2.out"
    seq2edits_out_file = "./outputs/seq2edits3.out" 
    m2_file = "./rule1_lm2_seq2edits3.m2"
    cmd = f"bash run_seq2edits.sh {lm_out_file} {seq2edits_out_file} {m2_file}"
    os.system(cmd)
    os.system(f"tail {m2_file}")
    print("Done!")

def test_lm1_seq2edits2():
    lm_file = "./outputs/lm.out"
    seq2edits_out_file = "./outputs/seq2edits2.out" 
    m2_file = "./lm1_seq2edits2.m2"
    cmd = f"bash run_seq2edits.sh {lm_file} {seq2edits_out_file} {m2_file}"
    os.system(cmd)
    os.system(f"tail {m2_file}")
    print("Done!")


if __name__ == "__main__":
    # test_model_serial()
    # test_model_parallel()
    test_rule1_lm2_seq2edits3()
    # test_lm1_seq2edits2()
