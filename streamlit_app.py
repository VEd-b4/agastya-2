import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# ===== DATA CLEANING FUNCTIONS (UNCHANGED) =====
def clean_and_process_data(df):
    """
    Clean and process student assessment data
    
    Parameters:
    df (pd.DataFrame): Raw dataframe from Excel
    
    Returns:
    pd.DataFrame: Cleaned and processed dataframe
    """
    
    initial_count = len(df)
    
    # ===== STEP 1: DATA CLEANING =====
    # Define pre and post question columns
    pre_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    post_questions = ['Q1_Post', 'Q2_Post', 'Q3_Post', 'Q4_Post', 'Q5_Post']
    
    # Condition 1: Remove rows where one set has values and the other is all NULL
    # If ANY pre question has values but ALL post questions are NULL
    has_any_pre = df[pre_questions].notna().any(axis=1)
    all_post_null = df[post_questions].isna().all(axis=1)
    remove_condition_1 = has_any_pre & all_post_null
    
    # If ALL pre questions are NULL but ANY post question has values
    all_pre_null = df[pre_questions].isna().all(axis=1)
    has_any_post = df[post_questions].notna().any(axis=1)
    remove_condition_2 = all_pre_null & has_any_post
    
    # Condition 3: Remove rows where BOTH pre and post are all NULL
    remove_condition_3 = all_pre_null & all_post_null
    
    # Remove rows that meet any of the conditions
    df = df[~(remove_condition_1 | remove_condition_2 | remove_condition_3)]
    
    cleaned_count = len(df)
    
    # ===== STEP 2: CALCULATE SCORES =====
    # Define answer columns
    pre_answers = ['Q1 Answer', 'Q2 Answer', 'Q3 Answer', 'Q4 Answer', 'Q5 Answer']
    post_answers = ['Q1_Answer_Post', 'Q2_Answer_Post', 'Q3_Answer_Post', 'Q4_Answer_Post', 'Q5_Answer_Post']
    
    # Calculate Pre-session scores
    df['Pre_Score'] = 0
    for q, ans in zip(pre_questions, pre_answers):
        df['Pre_Score'] += (df[q] == df[ans]).astype(int)
    
    # Calculate Post-session scores
    df['Post_Score'] = 0
    for q, ans in zip(post_questions, post_answers):
        df['Post_Score'] += (df[q] == df[ans]).astype(int)
    
    # ===== STEP 3: STANDARDIZE PROGRAM TYPES =====
    # Create a mapping for program types
    # SCB, SCC, SCM, SCP are combined into PCMB
    program_type_mapping = {
        'SCB': 'PCMB',
        'SCC': 'PCMB',
        'SCM': 'PCMB',
        'SCP': 'PCMB',
        'E-LOB': 'ELOB',
        'DLC-2': 'DLC',
        'DLC2': 'DLC'
    }
    
    # Apply the mapping
    df['Program Type'] = df['Program Type'].replace(program_type_mapping)
    
    # ===== STEP 4: CREATE PARENT CLASS =====
    # Extract parent class from Class column (e.g., "6-A" -> "6", "7-B" -> "7")
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    
    # Filter for grades 6-10 only
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]
    
    return df, initial_count, cleaned_count

# ===== TAB 8: SUBJECT ANALYSIS (MODIFIED) =====
def tab8_subject_analysis(df):
    """
    Generates the Subject-wise Performance and Participation Analysis, 
    including breakdowns by Region.
    """
    st.header("Subject-wise Performance Analysis")
    st.markdown("### Performance, Participation, and Assessment Count by Subject")
    
    # Check for required columns
    required_cols = ['Subject', 'Region']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Columns {required_cols} not found in the data. Cannot perform Subject-Region Analysis.")
        return

    # Calculate Subject Statistics (Overall)
    subject_stats = df.groupby('Subject').agg(
        Num_Students=('Student Id', 'nunique'),        # Number of unique students
        Num_Assessments=('Student Id', 'count'),       # Total number of assessments (rows)
        Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
        Avg_Post_Score_Raw=('Post_Score', 'mean')
    ).reset_index()

    # Calculate percentages and improvement
    subject_stats['Avg Pre Score %'] = (subject_stats['Avg_Pre_Score_Raw'] / 5) * 100
    subject_stats['Avg Post Score %'] = (subject_stats['Avg_Post_Score_Raw'] / 5) * 100
    subject_stats['Improvement %'] = subject_stats['Avg Post Score %'] - subject_stats['Avg Pre Score %']
    
    # Sort in ascending order of Post Score %
    subject_stats = subject_stats.sort_values('Avg Post Score %', ascending=True)
    
    # --- Visualization: Performance (Overall) ---
    st.subheader("📈 Overall Subject Performance Comparison (Pre vs. Post)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=subject_stats['Subject'],
        y=subject_stats['Avg Pre Score %'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_stats['Avg Pre Score %']],
        textposition='top center',
        textfont=dict(size=12, color='#3498db')
    ))
    
    fig.add_trace(go.Scatter(
        x=subject_stats['Subject'],
        y=subject_stats['Avg Post Score %'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_stats['Avg Post Score %']],
        textposition='top center',
        textfont=dict(size=12, color='#e74c3c')
    ))
    
    fig.update_layout(
        title='Subject-wise Pre and Post Assessment Scores (Overall)',
        xaxis_title='Subject',
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
    
    # Calculate Subject-Region Statistics for breakdowns
    subject_region_stats = df.groupby(['Subject', 'Region']).agg(
        Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
        Avg_Post_Score_Raw=('Post_Score', 'mean')
    ).reset_index()
    
    subject_region_stats['Pre_Score_Pct'] = (subject_region_stats['Avg_Pre_Score_Raw'] / 5) * 100
    subject_region_stats['Post_Score_Pct'] = (subject_region_stats['Avg_Post_Score_Raw'] / 5) * 100
    
    st.markdown("---")
    
    # ====================================================================
    # 1. NEW GRAPH: Subject Analysis per Region 
    # ====================================================================
    st.subheader("🗺️ Subject Performance within a Selected Region")
    
    unique_regions = sorted(df['Region'].unique())
    selected_region_for_subject = st.selectbox("Select Region for Subject Breakdown", unique_regions, key='region_subject_select')

    # Filter data for the selected region
    region_subject_data = subject_region_stats[subject_region_stats['Region'] == selected_region_for_subject].copy()
    region_subject_data = region_subject_data.sort_values('Post_Score_Pct', ascending=True)

    fig_subj_region = go.Figure()
    
    fig_subj_region.add_trace(go.Scatter(
        x=region_subject_data['Subject'],
        y=region_subject_data['Pre_Score_Pct'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#8e44ad', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in region_subject_data['Pre_Score_Pct']],
        textposition='top center'
    ))
    
    fig_subj_region.add_trace(go.Scatter(
        x=region_subject_data['Subject'],
        y=region_subject_data['Post_Score_Pct'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#f39c12', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in region_subject_data['Post_Score_Pct']],
        textposition='top center'
    ))
    
    fig_subj_region.update_layout(
        title=f'Subject Performance in **{selected_region_for_subject}** (Ascending by Post Score)',
        xaxis_title='Subject',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(gridcolor='#404040')
    )
    
    st.plotly_chart(fig_subj_region, use_container_width=True)
    
    st.markdown("---")

    # ====================================================================
    # 2. NEW GRAPH: Region Analysis per Subject 
    # ====================================================================
    st.subheader("📍 Region Performance for a Selected Subject")
    
    unique_subjects = sorted(df['Subject'].unique())
    selected_subject_for_region = st.selectbox("Select Subject for Region Breakdown", unique_subjects, key='subject_region_select')

    # Filter data for the selected subject
    subject_region_data = subject_region_stats[subject_region_stats['Subject'] == selected_subject_for_region].copy()
    subject_region_data = subject_region_data.sort_values('Post_Score_Pct', ascending=True)

    fig_region_subj = go.Figure()
    
    fig_region_subj.add_trace(go.Scatter(
        x=subject_region_data['Region'],
        y=subject_region_data['Pre_Score_Pct'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#1abc9c', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_region_data['Pre_Score_Pct']],
        textposition='top center'
    ))
    
    fig_region_subj.add_trace(go.Scatter(
        x=subject_region_data['Region'],
        y=subject_region_data['Post_Score_Pct'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_region_data['Post_Score_Pct']],
        textposition='top center'
    ))
    
    fig_region_subj.update_layout(
        title=f'Region Performance in **{selected_subject_for_region}** (Ascending by Post Score)',
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
    
    st.plotly_chart(fig_region_subj, use_container_width=True)
    
    # --- Detailed Table and Participation Metrics (UNCHANGED) ---
    st.markdown("---")
    st.subheader("📋 Subject Participation and Detailed Metrics (Overall)")

    # Create the display dataframe
    display_subject_stats = subject_stats.copy()
    display_subject_stats = display_subject_stats[[
        'Subject', 
        'Num_Students', 
        'Num_Assessments', 
        'Avg Pre Score %', 
        'Avg Post Score %', 
        'Improvement %'
    ]]
    
    display_subject_stats.columns = [
        'Subject', 
        'Unique Students', 
        'Total Assessments', 
        'Avg Pre %', 
        'Avg Post %', 
        'Improvement %'
    ]
    
    # Apply string formatting
    display_subject_stats['Avg Pre %'] = display_subject_stats['Avg Pre %'].apply(lambda x: f"{x:.1f}%")
    display_subject_stats['Avg Post %'] = display_subject_stats['Avg Post %'].apply(lambda x: f"{x:.1f}%")
    display_subject_stats['Improvement %'] = display_subject_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_subject_stats, hide_index=True, use_container_width=True)
    
    # --- Key Participation Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Subjects", len(subject_stats))
    with col2:
        st.metric("Total Unique Students Assessed", subject_stats['Num_Students'].sum())
    with col3:
        st.metric("Total Assessments Conducted", subject_stats['Num_Assessments'].sum())
        
    # Download Button
    st.markdown("---")
    subject_csv = display_subject_stats.to_csv(index=False)
    st.download_button(
        "📥 Download Subject Analysis Data (CSV)",
        subject_csv,
        "subject_analysis.csv",
        "text/csv"
    )
    
    return subject_stats # Return for use in the main download section

# ===== MAIN APPLICATION (MODIFIED TO FIX ERRORS) =====

# Title and description
st.title("📊 Student Assessment Analysis Platform")
st.markdown("### Upload, Clean, and Analyze Student Performance Data")

# File uploader
uploaded_file = st.file_uploader("Upload Student Data Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    
    # Load and clean data
    with st.spinner("Loading and cleaning data..."):
        try:
            raw_df = pd.read_excel(uploaded_file)
            
            # Basic checks for required columns (FIX 1: Corrected undefined variable required_df)
            required_check_cols = ['Date_Post', 'Donor', 'Subject', 'Region', 'Student Id', 'Class', 'Program Type', 'Q1', 'Q1_Post']
            missing_cols = [col for col in required_check_cols if col not in raw_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}. Please add these columns and try again.")
                st.stop()
                
            df, initial_count, cleaned_count = clean_and_process_data(raw_df)
            
            # Show cleaning summary
            st.success("✅ Data cleaned successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Records", initial_count)
            with col2:
                st.metric("Records Removed", initial_count - cleaned_count)
            with col3:
                st.metric("Final Records", cleaned_count)
            
            # Option to download cleaned data
            st.markdown("---")
            cleaned_excel = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Data (CSV)",
                data=cleaned_excel,
                file_name="cleaned_student_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
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
    
    # ===== KEY METRICS (MODIFIED SECTION) =====
    st.markdown("---")
    st.subheader("📊 Key Performance Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        avg_pre = (filtered_df['Pre_Score'].mean() / 5) * 100
        st.metric("Avg Pre Score", f"{avg_pre:.1f}%")
    
    with col2:
        avg_post = (filtered_df['Post_Score'].mean() / 5) * 100
        st.metric("Avg Post Score", f"{avg_post:.1f}%")
    
    with col3:
        improvement = avg_post - avg_pre
        st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    
    
    if not filtered_df.empty:
        # 1. Identify unique tests per student
        # A unique test is defined by Student, Content, Class, School AND Date.
        unique_student_tests = filtered_df.groupby(
            ['Student Id', 'Content Id', 'Class', 'School Name', 'Date_Post']
        ).size().reset_index(name='count')
        
        # 2. Count how many such unique tests each student has taken
        student_activity = unique_student_tests.groupby('Student Id').size().reset_index(name='Visible_Tests')

        # 3. Calculate metrics
        avg_tests = student_activity['Visible_Tests'].mean()
        max_tests = student_activity['Visible_Tests'].max()
        min_tests = student_activity['Visible_Tests'].min()
        
        # 4. NEW: Calculate test count distribution and percentages
        # Count the number of unique students for each test count
        test_counts = student_activity['Visible_Tests'].value_counts().reset_index()
        test_counts.columns = ['Tests Taken', 'Num Students']
        
        total_unique_students = student_activity['Student Id'].nunique()
        test_counts['Percentage'] = (test_counts['Num Students'] / total_unique_students) * 100
        
        # Sort in descending order of Tests Taken
        test_counts = test_counts.sort_values('Tests Taken', ascending=False)
        
    else:
        avg_tests, max_tests, min_tests = 0, 0, 0
        test_counts = pd.DataFrame() # Initialize empty dataframe
    
    with col4:
        st.metric("Avg Tests/Student", f"{avg_tests:.1f}")

    with col5:
        st.metric("Max Tests/Student", f"{max_tests}")
        
        # Display the distribution of test counts below the metric
        if not test_counts.empty:
            st.markdown("##### Test Count Distribution:")
            # Limit to top 10 counts for display brevity
            limit = min(10, len(test_counts)) 
            top_counts = test_counts.head(limit) 
            
            distribution_text = ""
            for index, row in top_counts.iterrows():
                # Format to show 'X times: Y.Y% (Z students)'
                distribution_text += f"* **{int(row['Tests Taken'])}x**: {row['Percentage']:.1f}% ({int(row['Num Students'])} students)\n"
                
            st.markdown(distribution_text)

    with col6:
        st.metric("Min Tests/Student", f"{min_tests}")
    
    # ===== TABS FOR DIFFERENT ANALYSES (UNCHANGED) =====
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📍 Region Analysis", "👤 Instructor Analysis", "📚 Grade Analysis", "📊 Program Type Analysis", "👥 Student Participation", "🏫 School Analysis", "💰 Donor Analysis", "🔬 Subject Analysis"])
    
    # Placeholder for subject_stats for download section
    subject_stats = None

    # ===== TAB 1: REGION ANALYSIS (UNCHANGED) =====
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
        
        # Sort in ascending order of Post Score Pct
        region_stats = region_stats.sort_values('Post_Score_Pct', ascending=True)
        
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
            title='Region-wise Performance Comparison (Ascending by Post Score)',
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
        # Note: This inner chart is not modified to ascending as it might break the context, 
        # but the request was for ALL line graphs, so I will apply the sort here as well.
        prog_data = prog_data.sort_values('Post_Score_Pct', ascending=True)

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
            title=f'{selected_program_type} - Region-wise Performance (Ascending by Post Score)',
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
    
    # ===== TAB 2: INSTRUCTOR ANALYSIS (UNCHANGED) =====
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
        
        # The general stats used for tables are sorted descending
        instructor_stats_for_table = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
        
        # Show top N instructors
        top_n = st.slider("Number of instructors to display", 5, 20, 10)
        
        # Get top N performers, then sort them ascending for the plot
        top_instructors = instructor_stats_for_table.nlargest(top_n, 'Post_Score_Pct').sort_values('Post_Score_Pct', ascending=True)

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
            title=f'Top {top_n} Instructors by Post-Session Performance (Ascending by Post Score)',
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
        
        # Instructor rankings (using the descending table stats)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Instructors")
            top_perf = instructor_stats_for_table.nlargest(10, 'Post_Score_Pct')[['Instructor Name', 'Post_Score_Pct', 'Student Id']]
            top_perf.columns = ['Instructor', 'Post Score %', 'Students']
            top_perf['Post Score %'] = top_perf['Post Score %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_perf, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("📈 Best Adaptation (Improvement)")
            best_adapt = instructor_stats_for_table.nlargest(10, 'Improvement')[['Instructor Name', 'Improvement', 'Student Id']]
            best_adapt.columns = ['Instructor', 'Improvement %', 'Students']
            best_adapt['Improvement %'] = best_adapt['Improvement %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(best_adapt, hide_index=True, use_container_width=True)
        
        # All Instructors Assessment Count
        st.markdown("---")
        st.subheader("📋 Complete Instructor List - Assessment Count")
        
        # FIX APPLIED: Correctly calculate assessments using Date_Post
        filtered_df['Assessment_Session_Key'] = (
            filtered_df['Content Id'].astype(str) + '_' + 
            filtered_df['Class'].astype(str) + '_' + 
            filtered_df['School Name'].fillna('NA').astype(str) + '_' +
            filtered_df['Date_Post'].astype(str) # Added Date_Post
        )
        
        # Calculate number of assessments (using the session key) per instructor
        all_instructors = filtered_df.groupby(['Instructor Name', 'Instructor Login Id']).agg({
            'Assessment_Session_Key': 'nunique', # This correctly counts distinct sessions
            'Student Id': 'count',
            'Region': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0] # Most common region
        }).reset_index()
        all_instructors.columns = ['Instructor Name', 'Instructor Login Id', 'Number of Assessments', 'Total Students', 'Primary Region']
        all_instructors = all_instructors.sort_values('Number of Assessments', ascending=False)
        
        # Search functionality
        search_instructor = st.text_input("🔍 Search for an instructor (Name or ID)", "")
        
        if search_instructor:
            filtered_instructors = all_instructors[
                all_instructors['Instructor Name'].str.contains(search_instructor, case=False, na=False) |
                all_instructors['Instructor Login Id'].astype(str).str.contains(search_instructor, case=False, na=False)
            ]
        else:
            filtered_instructors = all_instructors
            
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Instructors", len(all_instructors))
        with col2:
            st.metric("Avg Assessments per Instructor", f"{all_instructors['Number of Assessments'].mean():.1f}")
        with col3:
            st.metric("Max Assessments by One Instructor", all_instructors['Number of Assessments'].max())
            
        # Display the full table
        st.dataframe(
            filtered_instructors,
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Download option for instructor assessment data
        instructor_csv = all_instructors.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Instructor Assessment List",
            instructor_csv,
            "instructor_assessments.csv",
            "text/csv"
        )
        
        # Instructors per Region
        st.markdown("---")
        st.subheader("👥 Number of Instructors per Region")
        instructors_per_region = filtered_df.groupby('Region')['Instructor Name'].nunique().reset_index()
        instructors_per_region.columns = ['Region', 'Number of Instructors']
        instructors_per_region = instructors_per_region.sort_values('Number of Instructors', ascending=False)
        
        # Create bar chart
        fig_inst_region = go.Figure()
        fig_inst_region.add_trace(go.Bar(
            x=instructors_per_region['Region'],
            y=instructors_per_region['Number of Instructors'],
            marker_color='#3498db',
            text=instructors_per_region['Number of Instructors'],
            textposition='outside',
            textfont=dict(size=14)
        ))
        
        fig_inst_region.update_layout(
            title='Number of Instructors by Region',
            xaxis_title='Region',
            yaxis_title='Number of Instructors',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig_inst_region, use_container_width=True)
        
        # Display table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(instructors_per_region, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Unique Instructors", filtered_df['Instructor Name'].nunique())
            st.metric("Average per Region", f"{instructors_per_region['Number of Instructors'].mean():.1f}")
            
    # ===== TAB 3: GRADE ANALYSIS (UNCHANGED) =====
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
        
        # Sort in ascending order of Post Score Pct
        grade_stats = grade_stats.sort_values('Post_Score_Pct', ascending=True)

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
            title='Grade-wise Performance Comparison (Ascending by Post Score)',
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

    # ===== TAB 4: PROGRAM TYPE ANALYSIS (MODIFIED FILTER LOGIC) =====
    with tab4:
        st.header("Program Type Performance Analysis")
        
        # 1. Overall Program Type Performance (Existing)
        program_stats = filtered_df.groupby('Program Type').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
        program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
        program_stats['Improvement'] = program_stats['Post_Score_Pct'] - program_stats['Pre_Score_Pct']
        
        # Sort in ascending order of Post Score Pct for the bar chart
        program_stats = program_stats.sort_values('Post_Score_Pct', ascending=True)

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
            title='Program Type Performance Comparison (All Regions, Ascending by Post Score)',
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            barmode='group',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Program Type vs Region Breakdown (NEW)
        st.subheader("Program Type Performance by Region")
        
        # --- Filter by Program Type ---
        unique_program_types = sorted(filtered_df['Program Type'].unique())
        selected_program_type_for_region = st.selectbox("Select a Program Type for Region Breakdown", 
                                                        unique_program_types, key='program_region_breakdown')
                                                        
        program_region_stats = filtered_df[filtered_df['Program Type'] == selected_program_type_for_region].groupby('Region').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        # Sort by Post Score %
        program_region_stats = program_region_stats.sort_values('Post_Score_Pct', ascending=True)

        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=program_region_stats['Region'],
            y=program_region_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in program_region_stats['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.add_trace(go.Scatter(
            x=program_region_stats['Region'],
            y=program_region_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in program_region_stats['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.update_layout(
            title=f'Performance of {selected_program_type_for_region} by Region (Ascending by Post Score)',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")

        # 3. Program Type vs Grade Breakdown (NEW)
        st.subheader("Program Type Performance by Grade")

        # --- Filter by Program Type ---
        selected_program_type_for_grade = st.selectbox("Select a Program Type for Grade Breakdown", 
                                                       unique_program_types, key='program_grade_breakdown')
                                                        
        program_grade_stats = filtered_df[filtered_df['Program Type'] == selected_program_type_for_grade].groupby('Parent_Class').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        program_grade_stats['Pre_Score_Pct'] = (program_grade_stats['Pre_Score'] / 5) * 100
        program_grade_stats['Post_Score_Pct'] = (program_grade_stats['Post_Score'] / 5) * 100
        
        # Sort by Post Score %
        program_grade_stats = program_grade_stats.sort_values('Post_Score_Pct', ascending=True)

        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=program_grade_stats['Parent_Class'],
            y=program_grade_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#8e44ad', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in program_grade_stats['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig3.add_trace(go.Scatter(
            x=program_grade_stats['Parent_Class'],
            y=program_grade_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in program_grade_stats['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig3.update_layout(
            title=f'Performance of {selected_program_type_for_grade} by Grade (Ascending by Post Score)',
            xaxis_title='Grade',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Program Type statistics table (Overall)
        st.markdown("---")
        st.subheader("Detailed Program Type Statistics (Overall)")
        
        display_stats = program_stats.copy()
        display_stats.columns = ['Program Type', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_stats = display_stats[['Program Type', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        
        # Recalculate and format improvement correctly for display table
        display_stats['Improvement %'] = (display_stats['Post %'].str.replace('%', '').astype(float) - display_stats['Pre %'].str.replace('%', '').astype(float)).apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_stats, hide_index=True, use_container_width=True)
        
    # ===== TAB 5: STUDENT PARTICIPATION (UNCHANGED) =====
    with tab5:
        st.header("Student Participation and Test Frequency Analysis")
        
        if filtered_df.empty:
            st.warning("No data available after filtering to analyze student participation.")
        else:
            # Re-calculating student_activity just in case this tab is accessed first
            unique_student_tests = filtered_df.groupby(
                ['Student Id', 'Content Id', 'Class', 'School Name', 'Date_Post']
            ).size().reset_index(name='count')
            student_activity = unique_student_tests.groupby('Student Id').size().reset_index(name='Visible_Tests')
            
            total_students = student_activity['Student Id'].nunique()
            max_tests = student_activity['Visible_Tests'].max()
            avg_tests = student_activity['Visible_Tests'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Unique Students", total_students)
            with col2:
                st.metric("Max Tests Taken by 1 Student", max_tests)
            with col3:
                st.metric("Average Tests per Student", f"{avg_tests:.1f}")
                
            st.markdown("---")
            st.subheader("Distribution of Tests Taken per Student")
            
            # Count the number of unique students for each test count
            test_counts = student_activity['Visible_Tests'].value_counts().reset_index()
            test_counts.columns = ['Tests Taken', 'Num Students']
            
            # Calculate Percentage
            test_counts['Percentage'] = (test_counts['Num Students'] / total_students) * 100
            
            # Sort in ascending order of Tests Taken for the bar chart
            test_counts = test_counts.sort_values('Tests Taken', ascending=True)

            # Create bar chart for distribution
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=test_counts['Tests Taken'].astype(str) + 'x',
                y=test_counts['Percentage'],
                marker_color='#1abc9c',
                text=[f"{val:.1f}%" for val in test_counts['Percentage']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Percentage of Students by Number of Tests Taken',
                xaxis_title='Number of Tests Taken',
                yaxis_title='Percentage of Unique Students (%)',
                height=500,
                plot_bgcolor='#2b2b2b',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                yaxis=dict(range=[0, test_counts['Percentage'].max() * 1.1], gridcolor='#404040'),
                xaxis=dict(gridcolor='#404040')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Detailed Student Test Frequency Table")
            
            # Prepare data for table
            test_counts['Num Students'] = test_counts['Num Students'].astype(int)
            test_counts['Percentage'] = test_counts['Percentage'].apply(lambda x: f"{x:.1f}%")
            
            # Sort for table display (descending by tests taken)
            display_table = test_counts.sort_values('Tests Taken', ascending=False)

            st.dataframe(display_table, hide_index=True, use_container_width=True)

    # ===== TAB 6: SCHOOL ANALYSIS (UNCHANGED) =====
    with tab6:
        st.header("School-wise Performance Analysis")
        
        school_stats = filtered_df.groupby('School Name').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'nunique'
        }).reset_index()
        
        school_stats.columns = ['School Name', 'Pre Score', 'Post Score', 'Unique Students']
        school_stats['Pre Score %'] = (school_stats['Pre Score'] / 5) * 100
        school_stats['Post Score %'] = (school_stats['Post Score'] / 5) * 100
        school_stats['Improvement %'] = school_stats['Post Score %'] - school_stats['Pre Score %']
        
        school_stats = school_stats.sort_values('Post Score %', ascending=False)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Schools", school_stats['School Name'].nunique())
        with col2:
            st.metric("Avg Post Score", f"{school_stats['Post Score %'].mean():.1f}%")
        with col3:
            st.metric("Max Students in 1 School", school_stats['Unique Students'].max())
        with col4:
            st.metric("Min Students in 1 School", school_stats['Unique Students'].min())

        st.markdown("---")
        
        # --- Top Schools Table ---
        st.subheader("🏆 Top Performing Schools (Post Score)")
        
        top_n_schools = st.slider("Number of top schools to display", 5, 50, 10, key='top_schools_slider')
        
        top_schools_display = school_stats.nlargest(top_n_schools, 'Post Score %').copy()
        top_schools_display['Pre Score %'] = top_schools_display['Pre Score %'].apply(lambda x: f"{x:.1f}%")
        top_schools_display['Post Score %'] = top_schools_display['Post Score %'].apply(lambda x: f"{x:.1f}%")
        top_schools_display['Improvement %'] = top_schools_display['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            top_schools_display[['School Name', 'Unique Students', 'Post Score %', 'Pre Score %', 'Improvement %']],
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")

        # --- School Search ---
        st.subheader("🔍 Search and Compare School Performance")
        
        search_school = st.text_input("Search for a school (Name)", "")
        
        if search_school:
            filtered_schools = school_stats[
                school_stats['School Name'].str.contains(search_school, case=False, na=False)
            ].sort_values('Post Score %', ascending=False)
            
            filtered_schools_display = filtered_schools.copy()
            filtered_schools_display['Pre Score %'] = filtered_schools_display['Pre Score %'].apply(lambda x: f"{x:.1f}%")
            filtered_schools_display['Post Score %'] = filtered_schools_display['Post Score %'].apply(lambda x: f"{x:.1f}%")
            filtered_schools_display['Improvement %'] = filtered_schools_display['Improvement %'].apply(lambda x: f"{x:.1f}%")
            
            if not filtered_schools_display.empty:
                st.dataframe(
                    filtered_schools_display[['School Name', 'Unique Students', 'Post Score %', 'Pre Score %', 'Improvement %']],
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No schools found matching the search criteria.")
                
    # ===== TAB 7: DONOR ANALYSIS (UNCHANGED) =====
    with tab7:
        st.header("Donor-wise Performance Analysis")
        
        donor_stats = filtered_df.groupby('Donor').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        donor_stats['Pre_Score_Pct'] = (donor_stats['Pre_Score'] / 5) * 100
        donor_stats['Post_Score_Pct'] = (donor_stats['Post_Score'] / 5) * 100
        donor_stats['Improvement'] = donor_stats['Post_Score_Pct'] - donor_stats['Pre_Score_Pct']
        
        # Sort in ascending order of Post Score Pct
        donor_stats = donor_stats.sort_values('Post_Score_Pct', ascending=True)

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=donor_stats['Donor'],
            y=donor_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in donor_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        fig.add_trace(go.Scatter(
            x=donor_stats['Donor'],
            y=donor_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in donor_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        fig.update_layout(
            title='Donor-wise Performance Comparison (Ascending by Post Score)',
            xaxis_title='Donor',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Donor statistics table
        st.subheader("Detailed Donor Statistics")
        display_stats = donor_stats.copy()
        display_stats.columns = ['Donor', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_stats = display_stats[['Donor', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        display_stats['Pre %'] = display_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Post %'] = display_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Improvement %'] = display_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_stats, hide_index=True, use_container_width=True)

    # ===== TAB 8: SUBJECT ANALYSIS (MODIFIED) - Call the function =====
    with tab8:
        # The tab8_subject_analysis function is defined above
        subject_stats = tab8_subject_analysis(filtered_df)

    # ===== FOOTER AND DOWNLOAD SECTION (UNCHANGED) =====
    st.markdown("---")
    st.markdown("© 2025 Student Assessment Dashboard - Analysis Complete")
    
else:
    st.info("👆 Please upload an Excel file to start the analysis.")
    st.markdown("---")
    st.subheader("📋 Required Excel Columns")
    st.markdown("""
    Your Excel file must contain these columns:
    
    **Identification Columns:**
    - `Region` - Geographic region
    - `School Name` - Name of the school
    - `Donor` - The donor/partner associated with the record
    - **`Subject` - The subject name**
    - `UDISE` - School unique ID
    - `Student Id` - Unique student identifier
    - `Class` - Class with section (e.g., 6-A, 7-B)
    - `Instructor Name` - Name of instructor
    - `Instructor Login Id` - Login ID of instructor
    - `Program Type` - Program type code
    - `Content Id` - Assessment/Content Identifier
    - `Date_Post` - Assessment Date (Required for tracking unique tests)
    
    **Pre-Session (Questions & Answers):**
    - `Q1`, `Q2`, `Q3`, `Q4`, `Q5` - Student responses
    - `Q1 Answer`, `Q2 Answer`, `Q3 Answer`, `Q4 Answer`, `Q5 Answer` - Correct answers
    
    **Post-Session (Questions & Answers):**
    - `Q1_Post`, `Q2_Post`, `Q3_Post`, `Q4_Post`, `Q5_Post` - Student responses
    - `Q1_Answer_Post`, `Q2_Answer_Post`, `Q3_Answer_Post`, `Q4_Answer_Post`, `Q5_Answer_Post` - Correct answers
    """)