
class Config(object):
    # 是否开启 （这三个选项已经移动到前端）
    # rule_corrector_on = True
    # lm_corrector_on = True
    # seq2edits_corrector_on = True

    # rule_file = "./rules.txt" # 设置为""就是无规则文件
    rule_file = "" # 设置为""就是无规则文件

    lm_alpha = 2
    lm_name = "gpt2-medium"
    # assert lm_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]

    seq2edits_encoder = "xlnet"
    # assert seq2edits_encoder in ["xlnet", "roberta", "bert", "distilbert"]
