import streamlit as st
from transformers import pipeline

summary = st.text_input('Work summary to predict from', 'Two knights set off on an adventure.')

classifier = pipeline(model='zdreiosis/ff_analysis_4', return_all_scores=True)
labels = classifier([summary])

st.write('Predicted labels: ')
for x in labels[0]:
  st.write(x['label'] + ':', x['score'])

