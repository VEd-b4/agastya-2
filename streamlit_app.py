import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# Title and description
st.title("📊 Student Assessment Analysis Dashboard")
st.markdown("### Pre-Session vs Post-Session Performance Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload Cleaned Student Data (Excel)", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)
    
    # Ensure scores are calculated (in case raw data is uploaded)
    if 'Pre_Score' not in df.columns or 'Post_Score' not in df.columns:
        st.error("Please upload the cleaned data with Pre_Score and Post_Score columns.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Region filter
    all_regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", all_regions)
    
    # Program Type filter
    all_programs = ['All'] + sorted(df['Program Type'].unique().tolist())
    selected_program = st.sidebar.selectbox("Select Program Type", all_programs)
    
    # Parent Class filter
    all_classes = ['All'] + sorted(df['Parent_Class'].unique().tolist())
    selected_class = st.sidebar.selectbox("Select Grade", all_classes)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_program != 'All':
        filtered_df = filtered_df[filtered_df['Program Type'] == selected_program]
    if selected_class != 'All':
        filtered_df = filtered_df[filtered_df['Parent_Class'] == selected_class]
    
    # ===== KEY METRICS =====
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pre = (filtered_df['Pre_Score'].mean() / 5) * 100
        st.metric("Avg Pre-Session Score", f"{avg_pre:.1f}%")
    
    with col2:
        avg_post = (filtered_df['Post_Score'].mean() / 5) * 100
        st.metric("Avg Post-Session Score", f"{avg_post:.1f}%")
    
    with col3:
        improvement = avg_post - avg_pre
        st.metric("Overall Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    
    with col4:
        avg_tests = filtered_df['Test_Count'].mean()
        st.metric("Avg Tests per Student", f"{avg_tests:.1f}")
    
    # ===== TABS FOR DIFFERENT ANALYSES =====
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Region Analysis", "👤 Instructor Analysis", "📚 Grade Analysis", "📊 Program Type Analysis"])
    
    # ===== TAB 1: REGION ANALYSIS =====
    with tab1:
        st.header("Region-wise Performance Analysis")
        
        # Overall region analysis
        region_stats = filtered_df.groupby('Region').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        region_stats['Pre_Score_Pct'] = (region_stats['Pre_Score'] / 5) * 100
        region_stats['Post_Score_Pct'] = (region_stats['Post_Score'] / 5) * 100
        region_stats['Improvement'] = region_stats['Post_Score_Pct'] - region_stats['Pre_Score_Pct']
        region_stats = region_stats.sort_values('Region')
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=12, color='#2ecc71')
        ))
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=12, color='#e67e22')
        ))
        
        fig.update_layout(
            title='Region-wise Performance Comparison',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            hovermode='x unified',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Region by Program Type
        st.subheader("Region Analysis by Program Type")
        
        program_region_stats = filtered_df.groupby(['Region', 'Program Type']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        selected_program_type = st.selectbox("Select Program Type for Detailed View", 
                                             sorted(filtered_df['Program Type'].unique()))
        
        prog_data = program_region_stats[program_region_stats['Program Type'] == selected_program_type]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=prog_data['Region'],
            y=prog_data['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in prog_data['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.add_trace(go.Scatter(
            x=prog_data['Region'],
            y=prog_data['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in prog_data['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.update_layout(
            title=f'{selected_program_type} - Region-wise Performance',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Top performing and most improved regions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Scoring Regions (Post-Session)")
            top_scoring = region_stats.nlargest(5, 'Post_Score_Pct')[['Region', 'Post_Score_Pct']]
            top_scoring['Post_Score_Pct'] = top_scoring['Post_Score_Pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_scoring, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("📈 Most Improved Regions (Adaptation)")
            most_improved = region_stats.nlargest(5, 'Improvement')[['Region', 'Improvement']]
            most_improved['Improvement'] = most_improved['Improvement'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(most_improved, hide_index=True, use_container_width=True)
    
    # ===== TAB 2: INSTRUCTOR ANALYSIS =====
    with tab2:
        st.header("Instructor-wise Performance Analysis")
        
        instructor_stats = filtered_df.groupby('Instructor Name').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        instructor_stats['Pre_Score_Pct'] = (instructor_stats['Pre_Score'] / 5) * 100
        instructor_stats['Post_Score_Pct'] = (instructor_stats['Post_Score'] / 5) * 100
        instructor_stats['Improvement'] = instructor_stats['Post_Score_Pct'] - instructor_stats['Pre_Score_Pct']
        instructor_stats = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
        
        # Show top N instructors
        top_n = st.slider("Number of instructors to display", 5, 20, 10)
        top_instructors = instructor_stats.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in top_instructors['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in top_instructors['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Instructors by Post-Session Performance',
            xaxis_title='Instructor',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(tickangle=-45, gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Instructor rankings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Instructors")
            top_perf = instructor_stats.nlargest(10, 'Post_Score_Pct')[['Instructor Name', 'Post_Score_Pct', 'Student Id']]
            top_perf.columns = ['Instructor', 'Post Score %', 'Students']
            top_perf['Post Score %'] = top_perf['Post Score %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_perf, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("📈 Best Adaptation (Improvement)")
            best_adapt = instructor_stats.nlargest(10, 'Improvement')[['Instructor Name', 'Improvement', 'Student Id']]
            best_adapt.columns = ['Instructor', 'Improvement %', 'Students']
            best_adapt['Improvement %'] = best_adapt['Improvement %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(best_adapt, hide_index=True, use_container_width=True)
    
    # ===== TAB 3: GRADE ANALYSIS =====
    with tab3:
        st.header("Grade-wise Performance Analysis")
        
        grade_stats = filtered_df.groupby('Parent_Class').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        grade_stats['Pre_Score_Pct'] = (grade_stats['Pre_Score'] / 5) * 100
        grade_stats['Post_Score_Pct'] = (grade_stats['Post_Score'] / 5) * 100
        grade_stats['Improvement'] = grade_stats['Post_Score_Pct'] - grade_stats['Pre_Score_Pct']
        grade_stats = grade_stats.sort_values('Parent_Class')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#1abc9c', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        fig.update_layout(
            title='Grade-wise Performance Comparison',
            xaxis_title='Grade',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Grade statistics table
        st.subheader("Detailed Grade Statistics")
        display_stats = grade_stats.copy()
        display_stats.columns = ['Grade', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_stats = display_stats[['Grade', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        display_stats['Pre %'] = display_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Post %'] = display_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Improvement %'] = display_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_stats, hide_index=True, use_container_width=True)
    
    # ===== TAB 4: PROGRAM TYPE ANALYSIS =====
    with tab4:
        st.header("Program Type Performance Analysis")
        
        program_stats = filtered_df.groupby('Program Type').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
        program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
        program_stats['Improvement'] = program_stats['Post_Score_Pct'] - program_stats['Pre_Score_Pct']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Pre_Score_Pct'],
            name='Pre-Session',
            marker_color='#3498db',
            text=[f"{val:.0f}%" for val in program_stats['Pre_Score_Pct']],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Post_Score_Pct'],
            name='Post-Session',
            marker_color='#e74c3c',
            text=[f"{val:.0f}%" for val in program_stats['Post_Score_Pct']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Program Type Performance Comparison',
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            barmode='group',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 110], gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Program stats table
        st.subheader("Program Type Statistics")
        display_prog = program_stats.copy()
        display_prog.columns = ['Program', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_prog = display_prog[['Program', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        display_prog['Pre %'] = display_prog['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_prog['Post %'] = display_prog['Post %'].apply(lambda x: f"{x:.1f}%")
        display_prog['Improvement %'] = display_prog['Improvement %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_prog, hide_index=True, use_container_width=True)
    
    # ===== DOWNLOAD SECTION =====
    st.markdown("---")
    st.subheader("📥 Download Analysis Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_csv = region_stats.to_csv(index=False)
        st.download_button("Download Region Analysis", region_csv, "region_analysis.csv", "text/csv")
    
    with col2:
        instructor_csv = instructor_stats.to_csv(index=False)
        st.download_button("Download Instructor Analysis", instructor_csv, "instructor_analysis.csv", "text/csv")
    
    with col3:
        grade_csv = grade_stats.to_csv(index=False)
        st.download_button("Download Grade Analysis", grade_csv, "grade_analysis.csv", "text/csv")

else:
    st.info("👆 Please upload your cleaned student data Excel file to begin analysis.")
    st.markdown("""
    ### Instructions:
    1. First run the data cleaning script on your raw Excel file
    2. Upload the cleaned Excel file here
    3. Use the sidebar filters to explore different views
    4. Navigate through tabs to see different analyses
    5. Download analysis reports as needed
    """)