import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="Construction Cloud", layout="wide")

# Title
st.title("Construction Project Management")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Project Tracking", "Resource Management", "Analytics"])

# Project Tracking
if menu == "Project Tracking":
    st.header("Project Tracking")
    projects = pd.DataFrame({
        "Project Name": ["Building A", "Road Construction", "Bridge Project"],
        "Status": ["In Progress", "Completed", "Planning"],
        "Start Date": ["2023-01-15", "2022-09-01", "2023-05-20"],
        "End Date": ["2023-12-31", "2023-03-15", "2023-11-30"]
    })
    st.dataframe(projects)

# Resource Management
elif menu == "Resource Management":
    st.header("Resource Management")
    resources = pd.DataFrame({
        "Resource": ["Cement", "Steel", "Labor"],
        "Quantity": ["500 bags", "20 tons", "15 workers"],
        "Cost": ["$5,000", "$10,000", "$7,500"]
    })
    st.dataframe(resources)

# Analytics
elif menu == "Analytics":
    st.header("Project Analytics")
    st.write("Budget vs Actual Spending")
    budget_data = pd.DataFrame({
        "Category": ["Materials", "Labor", "Equipment"],
        "Budget": [50000, 30000, 20000],
        "Actual": [45000, 32000, 18000]
    })
    st.bar_chart(budget_data.set_index("Category"))