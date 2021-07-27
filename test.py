import sys
from corrector import RulesCorrector, LMCorrector, Seq2EditsCorrector, MultiCorrector
from config import Config

def test(method):
    examples = [
        "Use this text too see an few of of the problems that LanguageTool can detecd. ",
        "What do you thinks of grammar checkers? ",
        "Please not that they are not perfect. ",
        "Style issues get a blue marker: ",
        "It's 5 P.M. in the afternoon. ",
        "The weather was nice on Thursday, 27 June 2017."
    ]
    if method == "rules":
        corrector = RulesCorrector()
    elif method == "lm":
        corrector = LMCorrector("gpt2", 2)
    elif method == "seq2edits":
        corrector = Seq2EditsCorrector("bert")
    elif method == "multi":
        corrector = MultiCorrector(Config)


    for text in examples:
        res = corrector(text) 
        print(text)
        print(res)
        print("\n\n")


if __name__ == "__main__":
    method = sys.argv[1]
    test(method)
