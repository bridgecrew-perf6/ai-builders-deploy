import streamlit as st
from transformers import pipeline

summary = st.text_input('Work summary to predict from', 'Two knights set off on an adventure.')
st.write('Work summary: ', title)

#classifier = pipeline(model='zdreiosis/ff_analysis_4', return_all_scores=True)
#labels = classifier(['Two knights go on an adventure.'])

#st.write(labels)


