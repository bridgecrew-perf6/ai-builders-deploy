import streamlit as st
from transformers import pipeline

classifier = pipeline(model='zdreiosis/ff_analysis_4', return_all_scores=True)
labels = classifier(['Two knights go on an adventure.'])

st.write(labels)


