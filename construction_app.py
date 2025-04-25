import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import time

# App title and config
st.set_page_config(
    page_title="EcoBuild Optimizer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': 'https://example.com',
        'About': "### EcoBuild Optimizer\n\nAn AI-powered tool for sustainable construction planning"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stExpander {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load material data from CSV
@st.cache_data
def load_materials_data():
    try:
        return pd.read_csv('materials_data.csv')
    except FileNotFoundError:
        # Default data if CSV doesn't exist
        return pd.DataFrame({
            'Material': ['Concrete', 'Steel', 'Wood', 'Brick', 'Glass', 'Bamboo', 'Recycled Steel', 'Hempcrete', 'Cross-Laminated Timber'],
            'Unit Cost': [100, 150, 80, 120, 200, 90, 170, 110, 130],
            'Carbon Footprint': [120, 200, 50, 80, 150, 30, 60, 20, 40],
            'Use Case': ['Foundation', 'Structure', 'Interior', 'Exterior', 'Windows', 'Flooring', 'Structure', 'Walls', 'Structure'],
            'Sustainability Score': [3, 2, 4, 3, 2, 5, 4, 5, 5],
            'Recyclability': [0.3, 0.9, 0.7, 0.4, 0.8, 0.9, 1.0, 0.8, 0.9]
        })

materials_data = load_materials_data()

# AI Recommendation System
def get_ai_recommendations(project_type, budget, area, location):
    """Generate AI-powered material recommendations based on project parameters"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data for ML model
    X = materials_data[['Unit Cost', 'Carbon Footprint', 'Sustainability Score', 'Recyclability']]
    
    # Create target score combining cost and carbon footprint
    materials_data['BudgetScore'] = 1 / (materials_data['Unit Cost'] * area / budget)
    materials_data['CarbonScore'] = 1 / materials_data['Carbon Footprint']
    y = materials_data['BudgetScore'] * 0.6 + materials_data['CarbonScore'] * 0.4
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train regressor model (better suited for continuous targets)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Generate recommendations
    predictions = model.predict(X_scaled)
    top3_indices = np.argsort(predictions)[-3:][::-1]
    recommendations = materials_data.iloc[top3_indices]
    return recommendations

# Main app function

def update_project_progress(project_name, progress):
    """Update project progress in session state"""
    for i, p in enumerate(st.session_state.projects):
        if p['name'] == project_name:
            st.session_state.projects[i]['progress'] = progress
            break

def main():
    # Load demo projects on first run
    if 'projects' not in st.session_state:
        st.session_state.projects = [
            {
                'name': 'Eco-Residence',
                'type': 'Residential',
                'area': 2500,
                'budget': 150000,
                'location': 'Portland, OR',
                'materials': ['Cross-Laminated Timber', 'Hempcrete', 'Bamboo'],
                'progress': 35
            },
            {
                'name': 'Green Office',
                'type': 'Commercial',
                'area': 10000,
                'budget': 750000,
                'location': 'San Francisco, CA',
                'materials': ['Recycled Steel', 'Bamboo', 'Glass'],
                'progress': 65
            }
        ]
    
    # Navigation menu
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Creation", "Project Details", "3D Visualization", "Gantt Chart", "AI Recommendations"], key="nav_menu")
    
    if page == "Project Creation":
        project_creation_page()
    elif page == "Project Details":
        project_details_page()
    elif page == "3D Visualization":
        visualization_page()
    elif page == "Gantt Chart":
        gantt_chart_page()
    elif page == "AI Recommendations":
        ai_recommendations_page()

def project_creation_page():
    st.title("🏗️ EcoBuild Optimizer")
    st.markdown("Select materials for your construction project based on cost and environmental impact")
    
    # Project inputs
    with st.expander("Project Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project Name", key="project_name")
            project_type = st.selectbox("Construction Type", ["Residential", "Commercial"])
            project_area = st.number_input("Project Area (sqft)", min_value=100, value=1000, key="project_area_input", on_change=lambda: st.session_state.update(project_area_updated=True, budget_updated=True, materials_updated=True))
        with col2:
            budget = st.number_input("Estimated Budget ($)", min_value=1000, value=50000, key="budget_input", on_change=lambda: st.session_state.update(budget_updated=True, project_area_updated=True, materials_updated=True))
            location = st.text_input("Project Location", "New York, NY")
            
            # Real-time budget analysis
            if 'budget_updated' in st.session_state:
                st.subheader("Budget Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Budget", f"${budget:,.0f}")
                    
                    # Calculate material costs
                    if 'selected_materials' in st.session_state and st.session_state.selected_materials:
                        total_material_cost = sum(
                            materials_data[materials_data['Material'] == material]['Unit Cost'].values[0] * 
                            st.session_state.get(f"qty_{material}", 0)
                            for material in st.session_state.selected_materials
                        )
                        remaining_budget = budget - total_material_cost
                        
                        st.metric("Material Costs", f"${total_material_cost:,.0f}", 
                                delta=f"{total_material_cost/budget*100:.1f}% of budget")
                        st.metric("Remaining Budget", f"${remaining_budget:,.0f}", 
                                delta_color="inverse" if remaining_budget < 0 else "normal")
                
                with col2:
                    # Budget allocation pie chart
                    if 'selected_materials' in st.session_state and st.session_state.selected_materials:
                        cost_data = []
                        for material in st.session_state.selected_materials:
                            unit_cost = materials_data[materials_data['Material'] == material]['Unit Cost'].values[0]
                            quantity = st.session_state.get(f"qty_{material}", 0)
                            cost_data.append({
                                'Category': material,
                                'Amount': unit_cost * quantity
                            })
                        
                        if remaining_budget > 0:
                            cost_data.append({'Category': 'Remaining', 'Amount': remaining_budget})
                        
                        cost_df = pd.DataFrame(cost_data)
                        fig = px.pie(cost_df, values='Amount', names='Category', 
                                    title="Budget Allocation", hole=0.3)
                        st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Save Project"):
                # Check if project exists to update or create new
                existing_idx = next((i for i, p in enumerate(st.session_state.get('projects', [])) 
                                  if p['name'] == project_name), None)
                
                current_project = {
                    'name': project_name,
                    'type': project_type,
                    'area': project_area,
                    'budget': budget,
                    'location': location,
                    'materials': st.session_state.get('selected_materials', []),
                    'progress': 0  # Initialize progress
                }
                
                if 'projects' not in st.session_state:
                    st.session_state.projects = []
                
                if existing_idx is not None:
                    # Preserve existing progress when updating
                    current_project['progress'] = st.session_state.projects[existing_idx].get('progress', 0)
                    st.session_state.projects[existing_idx] = current_project
                    st.success(f"Project '{project_name}' updated!")
                else:
                    # Add new project
                    st.session_state.projects.append(current_project)
                    st.success(f"Project '{project_name}' saved!")
                
                # Clear form and force refresh
                st.session_state.project_name = ""
                st.rerun()

    # Material selection and cost calculator
    st.subheader("Material Selection & Cost Calculator")
    
    selected_materials = st.multiselect(
        "Choose construction materials",
        materials_data['Material'].unique(),
        default=["Concrete", "Steel"],
        key="material_selection",
        on_change=lambda: st.session_state.update(materials_updated=True, budget_updated=True)
    )
    
    # Quantity input for each material
    if selected_materials:
        with st.expander("Material Quantities", expanded=True):
            quantities = {}
            for material in selected_materials:
                quantities[material] = st.number_input(
                    f"Quantity for {material} (units)",
                    min_value=1,
                    value=100,
                    key=f"qty_{material}"
                )
    
    # Interactive cost breakdown
    if selected_materials:
        with st.expander("Cost Breakdown", expanded=True):
            cost_data = []
            for material in selected_materials:
                unit_cost = materials_data[materials_data['Material'] == material]['Unit Cost'].values[0]
                total_cost = unit_cost * quantities[material]
                cost_data.append({
                    'Material': material,
                    'Unit Cost': unit_cost,
                    'Quantity': quantities[material],
                    'Total Cost': total_cost
                })
            
            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df.style.highlight_max(axis=0, color='#ffcccb'))
            
            # Pie chart visualization
            fig = px.pie(cost_df, values='Total Cost', names='Material', 
                         title="Material Cost Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Filter materials based on selection
    filtered_data = materials_data[materials_data['Material'].isin(selected_materials)]
    
    # Calculations
    if not filtered_data.empty:
        filtered_data.loc[:, 'Total Cost'] = filtered_data['Unit Cost'] * project_area
        filtered_data.loc[:, 'Total CO2'] = filtered_data['Carbon Footprint'] * project_area
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(filtered_data.style.highlight_max(axis=0, color='#ffcccb'))
            
        with col2:
            fig = px.bar(
                filtered_data, 
                x='Material', 
                y='Total CO2',
                title="Carbon Footprint by Material",
                color='Total Cost',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive construction timeline
        st.subheader("Interactive Construction Timeline")
        
        phases = st.multiselect(
            "Select construction phases",
            ["Excavation", "Foundation", "Framing", "MEP", "Finishes"],
            default=["Foundation", "Framing"]
        )
        
        if phases:
            timeline_data = []
            for i, phase in enumerate(phases):
                timeline_data.append({
                    'Phase': phase,
                    'Start': f"2023-0{i+1}-01",
                    'End': f"2023-0{i+2}-15",
                    'Progress': min(100, (i+1)*25)
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            fig = px.timeline(
                timeline_df,
                x_start="Start",
                x_end="End",
                y="Phase",
                color="Progress",
                color_continuous_scale='Viridis',
                title="Construction Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Progress slider
            progress = st.slider("Adjust timeline progress", 0, 100, 50)
            st.write(f"Current progress: {progress}%")
        
        # Real-time updates
        with st.expander("Project Timeline"):
            timeline_data = pd.DataFrame({
                'Phase': ['Planning', 'Design', 'Construction', 'Completion'],
                'Start': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-06-01']),
                'End': pd.to_datetime(['2023-01-31', '2023-02-28', '2023-05-31', '2023-06-30']),
                'Progress': [100, 100, 75, 0]
            })
            
            fig = px.timeline(
                timeline_data, 
                x_start="Start", 
                x_end="End", 
                y="Phase",
                color="Progress",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            progress = st.slider("Construction Progress", 0, 100, 50)
            st.write(f"Current progress: {progress}%")
    
    # Export options
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            time.sleep(2)
            st.success("Report generated successfully!")
            
            # Enhanced report with AI insights
            report_data = filtered_data.copy()
i see            recommendations = get_ai_recommendations(project_type, budget, project_area, location)
            report_data['AI Recommendation'] = report_data['Material'].isin(recommendations['Material'])
            
            st.download_button(
                label="Download Report",
                data=report_data.to_csv().encode('utf-8'),
                file_name='ecobuild_report.csv',
                mime='text/csv'
            )
            
            # Carbon impact visualization
            st.subheader("Carbon Impact Analysis")
            fig = px.pie(report_data, names='Material', values='Total CO2', 
                         title="Carbon Footprint Distribution",
                         color='AI Recommendation',
                         color_discrete_map={True: 'green', False: 'red'})
            st.plotly_chart(fig, use_container_width=True)

    return project_type, project_area, budget, location, selected_materials

def project_details_page():
    st.title("Project Dashboard")
    
    # Real-time project monitoring
    if 'projects' in st.session_state and st.session_state.projects:
        st.subheader("Project Overview")
        
        # Create project selector
        project_idx = 0  # Initialize with default value
        selected_project = st.session_state.projects[0]['name'] if st.session_state.projects else ""
        selected_project = st.selectbox(
            "Choose a project",
            [p['name'] for p in st.session_state.projects],
            key="project_selector"
        )
        
        # Get selected project data
        project_idx = next(i for i, p in enumerate(st.session_state.projects) if p['name'] == selected_project)
        project = st.session_state.projects[project_idx]
        
        # Ensure project has progress value
        if 'progress' not in project:
            project['progress'] = 0
        
        # Editable project details
        with st.expander("Edit Project Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                project['name'] = st.text_input("Project Name", project['name'], key=f"edit_name_{project_idx}")
                project['type'] = st.selectbox("Construction Type", ["Residential", "Commercial"], 
                                            index=0 if project['type'] == "Residential" else 1, 
                                            key=f"edit_type_{project_idx}")
                project['area'] = st.number_input("Project Area (sqft)", min_value=100, 
                                               value=project['area'], key=f"edit_area_{project_idx}")
            with col2:
                project['budget'] = st.number_input("Estimated Budget ($)", min_value=1000, 
                                                  value=project['budget'], key=f"edit_budget_{project_idx}")
                project['location'] = st.text_input("Project Location", project['location'], 
                                                  key=f"edit_location_{project_idx}")
                
        # Project progress slider
        progress = st.slider(
            "Project Completion",
            0, 100, project['progress'],
            key=f"progress_{project['name']}",
            on_change=lambda: update_project_progress(project['name'], st.session_state[f"progress_{project['name']}"])
        )
        
        # Update session state with any changes
        st.session_state.projects[project_idx] = project
        
        # Create metrics columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Project Budget",
                value=f"${project['budget']:,.0f}",
                delta=f"{progress}% complete"
            )
            
        with col2:
            # Calculate material costs
            if project['materials']:
                total_cost = sum(
                    materials_data[materials_data['Material'] == material]['Unit Cost'].values[0] * 100
                    for material in project['materials']
                )
                st.metric(
                    "Estimated Cost",
                    f"${total_cost:,.0f}",
                    delta=f"{total_cost/project['budget']*100:.1f}% of budget"
                )
    
    # Project management section
    with st.expander("Project Management", expanded=True):
        project_name = st.text_input("Project Name", key="project_name")
        if st.button("Save Project"):
            current_project = {
                'name': project_name,
                'type': st.session_state.get('project_type', ''),
                'area': st.session_state.get('project_area', 0),
                'budget': st.session_state.get('budget', 0),
                'location': st.session_state.get('location', ''),
                'materials': st.session_state.get('selected_materials', []),
                'progress': st.session_state.get('progress', 0)
            }
            if 'projects' not in st.session_state:
                st.session_state.projects = []
            st.session_state.projects.append(current_project)
            st.success(f"Project '{project_name}' saved!")
            st.rerun()
    
    # Display saved projects with delete option
    if 'projects' in st.session_state and st.session_state.projects:
        st.subheader("Saved Projects")
        projects_df = pd.DataFrame(st.session_state.projects)
        
        # Add delete buttons for each project
        projects_df['Delete'] = False
        edited_df = st.data_editor(
            projects_df,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Delete",
                    help="Select projects to delete",
                    default=False,
                )
            },
            disabled=["materials"],  # Only materials column remains disabled
            hide_index=True,
        )
        
        if st.button("Delete Selected Projects"):
            # Get indices of projects to delete
            to_delete = edited_df[edited_df['Delete']].index
            # Remove from session state
            st.session_state.projects = [
                p for i, p in enumerate(st.session_state.projects) 
                if i not in to_delete
            ]
            st.rerun()
    
    # Enhanced project monitoring dashboard
    st.subheader("Project Dashboard")
    
    if 'projects' in st.session_state and st.session_state.projects:
        # Create project selector
        selected_project = st.selectbox(
            "Choose a project",
            [p['name'] for p in st.session_state.projects],
            key=f"project_selector_{selected_project}"
        )
        
        # Get selected project data
        project = next(p for p in st.session_state.projects if p['name'] == selected_project)
        
        # Create toggle-based dashboard
        tab1, tab2, tab3 = st.tabs(["Budget", "Progress", "Materials"])
        
        with tab1:
            # Budget visualization for selected project
            budget_df = pd.DataFrame({
                'Category': ['Budget', 'Estimated Cost'],
                'Amount': [
                    project['budget'],
                    sum(materials_data[materials_data['Material'] == material]['Unit Cost'].values[0] * 100
                    for material in project['materials']) if project['materials'] else 0
                ]
            })
            fig = px.bar(budget_df, x='Category', y='Amount', 
                        title=f"Budget Analysis for {project['name']}")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Progress visualization for selected project
            progress_df = pd.DataFrame({
                'Category': ['Progress', 'Target'],
                'Percentage': [project['progress'], 100]
            })
            fig = px.bar(progress_df, x='Category', y='Percentage', 
                        title=f"Progress Tracking for {project['name']}")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Material usage for selected project
            if project['materials']:
                material_df = pd.DataFrame({
                    'Material': project['materials'],
                    'Used': [1 for _ in project['materials']]
                })
                fig = px.pie(material_df, names='Material', values='Used', 
                            title=f"Materials for {project['name']}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No materials selected for this project")
    
    # Resource management
    st.subheader("Resource Management")
    resources = st.multiselect(
        "Select Resources to Allocate",
        ['Cement', 'Steel', 'Wood', 'Labor', 'Equipment'],
        default=['Cement', 'Steel']
    )
    
    if st.button("Generate Resource Allocation Plan"):
        with st.spinner("Generating plan..."):
            time.sleep(1)
            st.success("Resource allocation plan generated!")
            
            # Display resource allocation
            allocation_data = pd.DataFrame({
                'Resource': resources,
                'Quantity': [100, 50, 200, 10, 5][:len(resources)],
                'Cost': [5000, 7500, 4000, 20000, 15000][:len(resources)]
            })
            st.dataframe(allocation_data)


def visualization_page():
    st.title("3D Site Visualization")
    
    # CSV upload for site data
    uploaded_file = st.file_uploader("Upload Site Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            site_data = pd.read_csv(uploaded_file)
            
            # Basic validation for required columns
            required_cols = ['latitude', 'longitude', 'carbon_footprint']
            if all(col in site_data.columns for col in required_cols):
                
                # Create 3D visualization
                st.subheader("3D Carbon Footprint Visualization")
                
                # Calculate view state based on data
                avg_lat = site_data['latitude'].mean()
                avg_lon = site_data['longitude'].mean()
                
                view_state = pdk.ViewState(
                    latitude=avg_lat,
                    longitude=avg_lon,
                    zoom=14,
                    pitch=45,
                )
                
                # Create layer with color based on carbon footprint
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=site_data,
                    get_position='[longitude, latitude]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=100,
                    pickable=True
                )
                
                st.pydeck_chart(pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style='mapbox://styles/mapbox/light-v9',
                    tooltip={
                        'html': '<b>Carbon Footprint:</b> {carbon_footprint}',
                        'style': {
                            'color': 'white'
                        }
                    }
                ))
                
                # Carbon footprint analysis
                st.subheader("Carbon Footprint Analysis")
                fig = px.histogram(site_data, x='carbon_footprint', 
                                  title="Distribution of Carbon Footprint")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def gantt_chart_page():
    st.title("Multi-Project Timeline")
    
    if 'projects' not in st.session_state or not st.session_state.projects:
        st.warning("No projects found. Please create projects first.")
        return
    
    # Create Gantt data for all projects
    all_tasks = []
    
    for project in st.session_state.projects:
        # Base tasks for each project
        tasks = [
            [f"{project['name']} - Planning", '2023-01-01', '2023-01-31', 100, 'General'],
            [f"{project['name']} - Design", '2023-02-01', '2023-02-28', 100, 'General']
        ]
        
        # Add material-specific tasks
        if 'materials' in project and project['materials']:
            materials = project['materials']
            if 'Concrete' in materials:
                tasks.append([f"{project['name']} - Foundation", '2023-01-01', '2023-02-10', 100, 'Concrete'])
            if 'Steel' in materials:
                tasks.append([f"{project['name']} - Structure", '2023-02-15', '2023-03-15', 75, 'Steel'])
            if 'Brick' in materials or 'Wood' in materials:
                tasks.append([f"{project['name']} - Walls", '2023-03-20', '2023-04-05', 50, 'Brick/Wood'])
            if 'Wood' in materials:
                tasks.append([f"{project['name']} - Roof", '2023-04-10', '2023-04-30', 25, 'Wood'])
            tasks.append([f"{project['name']} - Finishing", '2023-05-01', '2023-06-15', 0, 'Paint'])
        
        all_tasks.extend(tasks)
    
    tasks_df = pd.DataFrame(all_tasks, columns=['Task', 'Start', 'End', 'Progress', 'Material'])
    
    # Interactive Gantt chart
    fig = px.timeline(
        tasks_df, 
        x_start="Start", 
        x_end="End", 
        y="Task",
        color="Progress",
        color_continuous_scale='Viridis',
        title="Construction Project Timeline"
    )
    
    # Add progress labels
    for i, task in enumerate(tasks_df.itertuples()):
        fig.add_annotation(
            x=pd.to_datetime(task.Start) + (pd.to_datetime(task.End) - pd.to_datetime(task.Start))/2,
            y=task.Task,
            text=f"{task.Progress}%",
            showarrow=False,
            font=dict(color="white")
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Task details
    selected_task = st.selectbox("Select Task", tasks_df['Task'])
    task_data = tasks_df[tasks_df['Task'] == selected_task].iloc[0]
    
    st.subheader(f"{selected_task} Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Start Date", pd.to_datetime(task_data.Start).strftime('%Y-%m-%d'))
        st.metric("End Date", pd.to_datetime(task_data.End).strftime('%Y-%m-%d'))
    
    with col2:
        st.metric("Progress", f"{task_data.Progress}%")
        st.metric("Primary Material", task_data.Material)


def ai_recommendations_page():
    st.title("AI Material Recommendations")
    
    # Project inputs
    with st.expander("Project Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            project_type = st.selectbox("Construction Type", ["Residential", "Commercial", "Industrial", "Infrastructure"])
            project_area = st.number_input("Project Area (sqft)", min_value=100, value=1000, key="project_area_input", on_change=lambda: st.session_state.update(project_area_updated=True, budget_updated=True, materials_updated=True))
        with col2:
            budget = st.number_input("Budget ($)", min_value=1000, value=50000)
            priority = st.select_slider("Priority", 
                                      options=["Cost", "Balanced", "Sustainability"],
                                      value="Balanced")
    
    # Generate recommendations
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing options..."):
            time.sleep(1)
            
            # Adjust weights based on priority
            if priority == "Cost":
                cost_weight = 0.8
                sustain_weight = 0.2
            elif priority == "Sustainability":
                cost_weight = 0.2
                sustain_weight = 0.8
            else:
                cost_weight = 0.5
                sustain_weight = 0.5
            
            # Get AI recommendations
            recommendations = get_ai_recommendations(project_type, budget, project_area, "")
            
            # Display results
            st.subheader("Recommended Materials")
            
            # Highlight best options
            highlight_cols = []
            if priority == "Cost":
                highlight_cols.append('Unit Cost')
            elif priority == "Sustainability":
                highlight_cols.append('Sustainability Score')
                highlight_cols.append('Carbon Footprint')
            
            st.dataframe(
                recommendations.style.highlight_min(subset=['Unit Cost'], color='lightgreen')
                              .highlight_max(subset=['Sustainability Score'], color='lightgreen')
                              .highlight_min(subset=['Carbon Footprint'], color='lightgreen')
            )
            
            # Show comparison
            st.subheader("Cost vs. Sustainability")
            fig = px.scatter(
                recommendations,
                x="Unit Cost",
                y="Sustainability Score",
                size="Carbon Footprint",
                color="Material",
                title="Material Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()