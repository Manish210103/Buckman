import streamlit as st
import subprocess
import streamlit as st
import subprocess

def run_python_file(filename):
    exec(open(filename).read())

st.title("Data Exploration")
st.write("High had the highest total Count of Risk Level at 281, followed by Low at 270 and Medium at 259.")
st.write("Count of Risk Level and Knowledge about Govt. Schemes are positively correlated with each other.")
st.write("4 Years to 6 Years in Risk Level High made up 7.90% of Count of Risk Level.")
st.write("Negative Return had the highest average Count of Return Earned at 30. More than 13 had the lowest Count of Return Earned at 23.20.")
image = st.image('powerBI_dashboard.jpg', caption='Image Caption')

if st.sidebar.button('Build Model'):
    subprocess.Popen(['streamlit', 'run', 'model.py'])
    st.write("Python file executed successfully!")

if st.sidebar.button('Recommendation'):
    subprocess.Popen(['streamlit', 'run', 'recommendation.py'])
    st.write("Python file executed successfully!")

