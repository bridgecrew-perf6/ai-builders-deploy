import streamlit as st
from transformers import pipeline

summary = st.text_input('Work summary to predict from', 'Two knights set off on an adventure.')

classifier = pipeline(model='zdreiosis/ff_analysis_4', return_all_scores=True)
labels = classifier([summary])

ksorted = [sorted(labels[y], key=lambda x: x['score'], reverse=True) for y in range(len(labels))]

st.write('Predicted labels: ')
for x in ksorted[0]:
  st.write(x['label'] + ':', x['score'])

