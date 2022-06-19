import streamlit as st
from transformers import pipeline

st.write('# FFnet work summary prediction')
summary = st.text_area('Work summary to predict from', value='Two knights set off on an adventure.')
predmodel = st.selectbox('model', ('zdreiosis/ff_analysis_5', 'zdreiosis/ff_analysis_4'))

if predmodel == 'zdreiosis/ff_analysis_4':
  revisionlist = ('998f684', '4c4e9b6', 'bae2482')
elif predmodel == 'zdreiosis/ff_analysis_5':
  revisionlist = ('1454370', '4a08027', '3f1f66b', 'b5f60f2', '1c7d5f4')

revision = st.selectbox('model version', revisionlist)
st.write('(Recommended setup: zdr5/1454370)')

classifier = pipeline(model=predmodel, revision=revision, return_all_scores=True)
labels = classifier([summary])

ksorted = [sorted(labels[y], key=lambda x: x['score'], reverse=True) for y in range(len(labels))]


st.write('Predicted labels: ')
for x in ksorted[0]:
  st.write(x['label'] + ':', x['score'])
  
st.write('--')
st.write('check out the [blog post!](https://medium.com/@syntrp2/nlp-for-genre-predictions-on-ffnet-an-antithesis-to-utilitarianism-4380524ca1fc)')

