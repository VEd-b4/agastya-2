    import pandas as pd
import numpy as np

def clean_and_process_data(excel_file_path):
    """
    Clean and process student assessment data from Excel file
    
    Parameters:
    excel_file_path (str): Path to the Excel file
    
    Returns:
    pd.DataFrame: Cleaned and processed dataframe
    """
    
    # Read the Excel file
    df = pd.read_excel(excel_file_path)
    
    print(f"Initial records: {len(df)}")
    
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
    
    # Remove rows that meet either condition
    df = df[~(remove_condition_1 | remove_condition_2)]
    
    print(f"Records after cleaning: {len(df)}")
    
    # ===== STEP 2: CALCULATE SCORES =====
    
    # Define answer columns
    pre_answers = ['Q1_Answer', 'Q2_Answer', 'Q3_Answer', 'Q4_Answer', 'Q5_Answer']
    post_answers = ['Q1_Post_Answer', 'Q2_Post_Answer', 'Q3_Post_Answer', 'Q4_Post_Answer', 'Q5_Post_Answer']
    
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
    program_type_mapping = {
        'SC': 'PCMB',
        'SC2': 'PCMB',
        'SCB': 'PCMB',
        'SCC': 'PCMB',
        'SCM': 'PCMB',
        'SCP': 'PCMB',
        'E-LOB': 'ELOB',
        'DLC-2': 'DLC'
    }
    
    # Apply the mapping
    df['Program Type'] = df['Program Type'].replace(program_type_mapping)
    
    # ===== STEP 4: CREATE PARENT CLASS =====
    
    # Extract parent class from Class column (e.g., "6-A" -> "6", "7-B" -> "7")
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    
    # Filter for grades 6-10 only
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]
    
    # ===== STEP 5: CALCULATE TEST FREQUENCY =====
    
    # Count how many times each student has taken tests
    df['Test_Count'] = df.groupby('Student Id')['Student Id'].transform('count')
    
    print(f"Final records: {len(df)}")
    print(f"\nProgram Types: {df['Program Type'].unique()}")
    print(f"Parent Classes: {sorted(df['Parent_Class'].unique())}")
    print(f"Regions: {df['Region'].unique()}")
    
    return df


def save_cleaned_data(df, output_path='cleaned_data.xlsx'):
    """Save cleaned data to Excel file"""
    df.to_excel(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")


# Example usage:
if __name__ == "__main__":
    # Replace with your Excel file path
    input_file = "student_data.xlsx"
    
    # Clean and process the data
    cleaned_df = clean_and_process_data(input_file)
    
    # Save cleaned data
    save_cleaned_data(cleaned_df, 'cleaned_student_data.xlsx')
    
    # Display summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nAverage Pre-Score: {cleaned_df['Pre_Score'].mean():.2f} / 5")
    print(f"Average Post-Score: {cleaned_df['Post_Score'].mean():.2f} / 5")
    print(f"Average Improvement: {(cleaned_df['Post_Score'] - cleaned_df['Pre_Score']).mean():.2f}")
    print(f"\nAverage number of tests per student: {cleaned_df['Test_Count'].mean():.2f}")
    print(f"Total unique students: {cleaned_df['Student Id'].nunique()}")