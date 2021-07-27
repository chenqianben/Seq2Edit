
from lm_corrector import LMCorrector_

def test():
    examples = [
        "Use this text too see an few of of the problems that LanguageTool can detecd. ",
        "What do you thinks of grammar checkers? ",
        "Please not that they are not perfect. ",
        "Style issues get a blue marker: ",
        "It's 5 P.M. in the afternoon. ",
        "The weather was nice on Thursday, 27 June 2017."
    ]
    corrector = LMCorrector_("gpt2", 2)


    for text in examples:
        res = corrector(text) 
        print(text)
        print(res)
        import ipdb;ipdb.set_trace()
        print(res["words_annotated"])
        print("\n\n")


if __name__ == "__main__":
    test()
    
