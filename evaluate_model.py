import csv
from datetime import datetime
import openai
import logging
import os
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-[...]")  # Replace with environment variable or secure storage
# openai.api_base = "http://localhost:1234/v1" # FOR LM STUDIO 

model_name = "gpt-4o-mini"

def generate_model_responses(system, instruction):
    """
    Sends a system and user instruction to the OpenAI model and returns the response.

    Parameters:
        system (str): The system message to set the context.
        instruction (str): The user instruction to process.

    Returns:
        str: The model's response or an error message if the request fails.
    """
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction}
            ],
            
            temperature = 0.01,
            #max_tokens= -1, # FOR LM STUDIO 
            #stream=false, # FOR LM STUDIO 
            top_p=0.1,
            frequency_penalty=1,
            presence_penalty=1,
            stop=["<|im_end|>"] # FOR PHI ["<|end|>"]
        )
        message = completion.choices[0].message['content']
        return message
    except Exception as e:
        logging.error(f"Error generating Response for instruction '{instruction}': {e}")
        return "Error generating response"

def process_csv_and_generate_output(input_csv_path, output_csv_path):
    """
    Processes the input CSV by sending each row's instruction to the OpenAI model
    and writing the results to the output CSV with an added 'rejected' column.

    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to the output CSV file.
    """
    try:
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile, delimiter=';')  # Specify semicolon as the delimiter
            # Standardize column names by stripping leading/trailing spaces
            fieldnames = [name.strip() for name in reader.fieldnames]
            
            # Print the actual field names to inspect them
            print("CSV Headers:", fieldnames)
            
            # Check if 'Instruction' and 'System' columns exist
            if 'Instruction' not in fieldnames or 'System' not in fieldnames:
                logging.error(f"Input CSV must contain 'Instruction' and 'System' columns. Found columns: {fieldnames}")
                return
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames + ['model_generated_output'], delimiter=';')
            writer.writeheader()
            
            for row in reader:
                # Strip whitespace from column names for safe access
                instruction = row.get('Instruction'.strip())
                system = row.get('System'.strip())
                
                if not instruction or not system:
                    logging.warning(f"Missing 'Instruction' or 'System' in row: {row}")
                    row['model_generated_output'] = 'Error: Missing data'
                else:
                    message = generate_model_responses(system, instruction)
                    row['model_generated_output'] = message if message else 'Error generating response'
                
                writer.writerow(row)
                logging.info(f"Processed row with Instruction ID: {row.get('ID', 'N/A')}")
                
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV: {e}")


if __name__ == "__main__":
    # Define the input and output directories
    input_dir = 'prepared'
    output_dir = 'evaluated'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each CSV file in the prepared folder
    for input_csv_path in glob.glob(os.path.join(input_dir, '*.csv')):
        # Define the output CSV path
        # Replace 'prepared_' with 'evaluated_' in the output file name
        output_csv_filename = os.path.basename(input_csv_path).replace('prepared_', 'evaluated_')
        output_csv_path = os.path.join(output_dir, output_csv_filename)
        
        # Call the function to process the CSV and generate the output
        process_csv_and_generate_output(input_csv_path, output_csv_path)
        
        print(f"Processed {input_csv_path} and saved to {output_csv_path}")
