import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pydeck as pdk

# Set page config
st.set_page_config(page_title="Construction Cloud", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("Construction Project Management")
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #1e90ff, #00bfff);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Project Tracking", "Resource Management", "Analytics", "3D Visualization"])

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
    
    # Interactive Gantt chart
    st.subheader("Project Timeline")
    fig = px.timeline(projects, 
                     x_start="Start Date", 
                     x_end="End Date", 
                     y="Project Name", 
                     color="Status",
                     title="Project Gantt Chart")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    
    # Project progress
    st.subheader("Project Progress")
    progress = st.slider("Update project progress", 0, 100, 50)
    st.progress(progress)

# Resource Management
elif menu == "Resource Management":
    st.header("Resource Management")
    resources = pd.DataFrame({
        "Resource": ["Cement", "Steel", "Labor"],
        "Quantity": ["500 bags", "20 tons", "15 workers"],
        "Cost": ["$5,000", "$10,000", "$7,500"]
    })
    st.dataframe(resources)
    
    # Resource allocation pie chart
    st.subheader("Resource Allocation")
    fig = px.pie(resources, values="Cost", names="Resource", 
                title="Resource Cost Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource drag and drop
    st.subheader("Resource Allocation")
    st.write("Drag resources to allocate them to projects")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Available Resources")
        for resource in resources["Resource"]:
            st.write(f"📦 {resource}")
    with col2:
        st.write("Allocated to Projects")
        st.write("Drop here")

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
    
    # Interactive budget explorer
    st.subheader("Budget Explorer")
    selected_category = st.selectbox("Select Category", budget_data["Category"])
    category_data = budget_data[budget_data["Category"] == selected_category]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Budget", f"${category_data['Budget'].values[0]:,}")
    with col2:
        st.metric("Actual", f"${category_data['Actual'].values[0]:,}")
    
    # Variance calculation
    variance = (category_data['Actual'].values[0] - category_data['Budget'].values[0]) / category_data['Budget'].values[0] * 100
    st.metric("Variance", f"{variance:.2f}%", delta_color="inverse")

# 3D Visualization
elif menu == "3D Visualization":
    st.header("3D Construction Visualization")
    
    # Sample 3D construction site data
    construction_data = pd.DataFrame({
        "lat": [28.6139, 28.6140, 28.6141],
        "lon": [77.2090, 77.2091, 77.2092],
        "name": ["Foundation", "Structure", "Finishing"],
        "progress": [80, 50, 20]
    })
    
    # 3D map visualization
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=28.6139,
            longitude=77.2090,
            zoom=16,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "ColumnLayer",
                data=construction_data,
                get_position="[lon, lat]",
                get_elevation="progress*10",
                get_fill_color="[255, 165 * (1 - progress/100), 0]",
                radius=20,
                elevation_scale=10,
                pickable=True,
                auto_highlight=True,
            ),
        ],
        tooltip={
            "html": "<b>Phase:</b> {name}<br/><b>Progress:</b> {progress}%",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    ))
    
    # Interactive budget explorer
    st.subheader("Budget Explorer")
    selected_category = st.selectbox("Select Category", budget_data["Category"])
    category_data = budget_data[budget_data["Category"] == selected_category]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Budget", f"${category_data['Budget'].values[0]:,}")
    with col2:
        st.metric("Actual", f"${category_data['Actual'].values[0]:,}")
    
    # Variance calculation
    variance = (category_data['Actual'].values[0] - category_data['Budget'].values[0]) / category_data['Budget'].values[0] * 100
    st.metric("Variance", f"{variance:.2f}%", delta_color="inverse")
    
    # Interactive budget explorer
    st.subheader("Budget Explorer")
    selected_category = st.selectbox("Select Category", budget_data["Category"])
    category_data = budget_data[budget_data["Category"] == selected_category]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Budget", f"${category_data['Budget'].values[0]:,}")
    with col2:
        st.metric("Actual", f"${category_data['Actual'].values[0]:,}")
    
    # Variance calculation
    variance = (category_data['Actual'].values[0] - category_data['Budget'].values[0]) / category_data['Budget'].values[0] * 100
    st.metric("Variance", f"{variance:.2f}%", delta_color="inverse")
    
    # Interactive budget explorer
    st.subheader("Budget Explorer")
    selected_category = st.selectbox("Select Category", budget_data["Category"])
    category_data = budget_data[budget_data["Category"] == selected_category]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Budget", f"${category_data['Budget'].values[0]:,}")
    with col2:
        st.metric("Actual", f"${category_data['Actual'].values[0]:,}")
    
    # Variance calculation
    variance = (category_data['Actual'].values[0] - category_data['Budget'].values[0]) / category_data['Budget'].values[0] * 100
    st.metric("Variance", f"{variance:.2f}%", delta_color="inverse")