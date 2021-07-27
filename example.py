import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.pipelines import pipeline

st.cache(show_spinner=False)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    nlp_pipe = pipeline('question-answering',model=model,tokenizer=tokenizer)
    return nlp_pipe

npl_pipe = load_model()
st.header("Prototyping an NLP solution")
st.text("This demo uses a model for Question Answering.")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Just some random text.")

question = st.text_input(label='Insert a question.')
text = st.text_area(label="Context")
if (not len(text)==0) and not (len(question)==0):
    x_dict = npl_pipe(context=text,question=question
    st.text('Answer: ',x_dict['answer'])
