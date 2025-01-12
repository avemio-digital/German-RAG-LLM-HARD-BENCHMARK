import csv
import glob
import re
import pandas as pd
import os

# Define file name lists
special_file_names = [
    "evaluated_hard-qa-with-multiple-references_test.csv", 
]

normal_file_names = [
    "evaluated_hard-reasoning-de_test.csv",
    "evaluated_hard-reasoning-en_test.csv",
    "evaluated_summarize-meeting-attendee-topic_test.csv",
    "evaluated_summarize-meeting-topic_test.csv"
]

# Regular expressions for extracting integers
qa_reference_pattern = r'\[(\d+)\]'

# Create directory for prepared normal files
os.makedirs('extracted', exist_ok=True)

# Process special files
for file_name in special_file_names:
    # Read the CSV file
    df = pd.read_csv(file_name, sep=';')

    # Initialize new columns
    df['model_References'] = df['model_generated_output'].apply(lambda x: re.findall(qa_reference_pattern, x))
    

    # Save the modified DataFrame to a new CSV file
    new_file_name = os.path.join('extracted', f"extracted_{file_name}")
    df.to_csv(new_file_name, sep=';', index=False, encoding='utf-8')
    print(f"Saved {new_file_name}")

# Process normal files
for file_name in normal_file_names:
    # Read the CSV file
    df = pd.read_csv(file_name, sep=';')

    # Save the DataFrame to the prepared folder with a new name
    new_file_name = os.path.join('extracted', f"extracted_{file_name}")
    df.to_csv(new_file_name, sep=';', index=False, encoding='utf-8')
    print(f"Saved {new_file_name}")