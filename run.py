import streamlit as st
from annotated_text import annotated_text

from corrector import RulesCorrector

def main():
    rule_corrector = RulesCorrector(rule_file="./rules.txt")

    st.title('英文文本校对测试系统')
    st.header('英文文本校对测试系统')
    st.subheader('一些可用的例子：')
    examples = [
        "Use this text too see an few of of the problems that LanguageTool can detecd. ",
        "What do you thinks of grammar checkers? ",
        "Please not that they are not perfect. ",
        "Style issues get a blue marker: ",
        "It's 5 P.M. in the afternoon. ",
        "The weather was nice on Thursday, 27 June 2017."
    ]
    for example in examples:
        st.text(example)
    
    text = st.text_area("请输入待校对的文本：", examples[0])
    if st.button("开始校对"):
        st.write("规则模型输入:"+text)
        tokens = rule_corrector(text)
        annotated_text(*tokens)
        annotated_text(
            "This ",
            ("is", "KEEP", "#8ef"),
            " some ",
            ("annotated", "DELETE", "#faa"),
            ("text", "noun", "#afa"),
            " for those of ",
            ("you", "pronoun", "#fea"),
            " who ",
            ("like", "verb", "#8ef"),
            " this sort of ",
            ("thing", "noun", "#afa"),
        )

rule_corrector = RulesCorrector(rule_file="./rules.txt")

st.title('英文文本校对测试系统')
st.header('英文文本校对测试系统')
st.subheader('一些可用的例子：')
examples = [
    "Use this text too see an few of of the problems that LanguageTool can detecd. ",
    "What do you thinks of grammar checkers? ",
    "Please not that they are not perfect. ",
    "Style issues get a blue marker: ",
    "It's 5 P.M. in the afternoon. ",
    "The weather was nice on Thursday, 27 June 2017."
]
for example in examples:
    st.text(example)

text = st.text_area("请输入待校对的文本：", examples[0])
if st.button("开始校对"):
    st.write("规则模型输入:"+text)
    tokens = rule_corrector(text)
    annotated_text(*tokens)
    annotated_text(
        "This ",
        ("is", "KEEP", "#8ef"),
        " some ",
        ("annotated", "DELETE", "#faa"),
        ("text", "noun", "#afa"),
        " for those of ",
        ("you", "pronoun", "#fea"),
        " who ",
        ("like", "verb", "#8ef"),
        " this sort of ",
        ("thing", "noun", "#afa"),
    )

# if __name__ == "__main__":
#     main()
