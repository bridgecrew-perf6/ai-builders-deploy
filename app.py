import streamlit as st
from transformers import pipeline

summary = st.text_area('Work summary to predict from', value='Two knights set off on an adventure.')
predmodel = st.selectbox(['zdreiosis/ff_analysis_5', 'zdreiosis/ff_analysis_4'])
classifier = pipeline(model=predmodel, return_all_scores=True)
labels = classifier([summary])

ksorted = [sorted(labels[y], key=lambda x: x['score'], reverse=True) for y in range(len(labels))]

st.write('Predicted labels: ')
for x in ksorted[0]:
  st.write(x['label'] + ':', x['score'])

