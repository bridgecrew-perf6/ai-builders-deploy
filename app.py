import streamlit as st
import numpy as np
import pandas as pd
import transformers
from transformers import pipeline
import torch
cuda_id = torch.cuda.current_device()

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.write(cuda_id)

classifier = pipeline(model='zdreiosis/ff_analysis_4', return_all_scores=True, device=cuda_id)
