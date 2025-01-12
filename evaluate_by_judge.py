import csv
from datetime import datetime
import openai
import logging
import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-[...]")  # Replace with environment variable or secure storage
# openai.api_base = "http://localhost:1234/v1" # FOR LM STUDIO 

model_name = "gpt-4o-mini"

def is_number(value):
    return isinstance(value, (int, float))


def generate_model_responses(system, instruction, response, model_generated_output):
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
                {"role": "system", "content": "You should respond in JSON format with the following keys: 'reasoning_of_metrics_and_correctness', 'language_quality', 'overall_correctness', 'instruction_following', 'overall_score'. The values should be a number between 0 and 100 except for the reasoning - here you should think about each metric and compare the target response and the generated response carefully together with the given context. You have to do your reasoning in english language."},
                {"role": "user", "content": "System Instruction\n" + system + "\nUser Instruction:\n"+ instruction + "\nTarget Response that is the wanted generated Answer should be seen as ground thruth" + response + "\nModel generated Response that should be evaluated against the system and user instructions and should be compared to the Target Response:\n" + model_generated_output}
            ],
            
            temperature = 0.01,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"}
        )
        message = completion.choices[0].message['content']
        
        # Parse the JSON content
        response_data = json.loads(message)
        
        # Extract each key separately
        reasoning_of_metrics_and_correctness = response_data.get('reasoning_of_metrics_and_correctness', None)

        language_quality = response_data.get('language_quality', None)
        language_quality = language_quality if is_number(language_quality) else None

        overall_correctness = response_data.get('overall_correctness', None)
        overall_correctness = overall_correctness if is_number(overall_correctness) else None

        instruction_following = response_data.get('instruction_following', None)
        instruction_following = instruction_following if is_number(instruction_following) else None

        overall_score = response_data.get('overall_score', None)
        overall_score = overall_score if is_number(overall_score) else None

        # Return the extracted values as a dictionary
        return {
            'reasoning_of_metrics_and_correctness': reasoning_of_metrics_and_correctness,
            'language_quality': language_quality,
            'overall_correctness': overall_correctness,
            'instruction_following': instruction_following,
            'overall_score': overall_score
        }
    except Exception as e:
        logging.error(f"Error generating Response for instruction '{instruction}': {e}")
        return {
            'reasoning_of_metrics_and_correctness': None,
            'language_quality': None,
            'overall_correctness': None,
            'instruction_following': None,
            'overall_score': None
        }


def generate_model_responses_reasoning(system, instruction, response, model_generated_output):
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
                {"role": "system", "content": "You should respond in JSON format with the following keys: 'reasoning_of_metrics_and_correctness', 'language_quality', 'overall_correctness', 'instruction_following', 'constrains_adherence', 'logical_consistency', 'final_solution_correctness', 'overall_score'. The values should be a number between 0 and 100. EXCEPT For the reasoning_of_metrics_and_correctness - here you should think about each metric and compare the target response and the generated response carefully together with the given context and present your thinking as a text. You have to do your reasoning in english language."},
                {"role": "user", "content": "System Instruction\n" + system + "\nUser Instruction:\n"+ instruction + "\nTarget Response that is the wanted generated Answer should be seen as ground thruth" + response + "\nModel generated Response that should be evaluated against the system and user instructions and should be compared to the Target Response:\n" + model_generated_output}
            ],
            
            temperature = 0.01,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"}
        )
        message = completion.choices[0].message['content']
        
        # Parse the JSON content
        response_data = json.loads(message)

        # Extract each key separately
        reasoning_of_metrics_and_correctness = response_data.get('reasoning_of_metrics_and_correctness', None)

        constrains_adherence = response_data.get('constrains_adherence', None)
        constrains_adherence = constrains_adherence if is_number(constrains_adherence) else None

        logical_consistency = response_data.get('logical_consistency', None)
        logical_consistency = logical_consistency if is_number(logical_consistency) else None

        final_solution_correctness = response_data.get('final_solution_correctness', None)
        final_solution_correctness = final_solution_correctness if is_number(final_solution_correctness) else None        

        language_quality = response_data.get('language_quality', None)
        language_quality = language_quality if is_number(language_quality) else None

        overall_correctness = response_data.get('overall_correctness', None)
        overall_correctness = overall_correctness if is_number(overall_correctness) else None

        instruction_following = response_data.get('instruction_following', None)
        instruction_following = instruction_following if is_number(instruction_following) else None

        overall_score = response_data.get('overall_score', None)
        overall_score = overall_score if is_number(overall_score) else None

        # Return the extracted values as a dictionary
        return {
            'reasoning_of_metrics_and_correctness': reasoning_of_metrics_and_correctness,
            'language_quality': language_quality,
            'overall_correctness': overall_correctness,
            'instruction_following': instruction_following,
            'constrains_adherence': constrains_adherence,
            'logical_consistency': logical_consistency,
            'final_solution_correctness': final_solution_correctness,
            'overall_score': overall_score
        }
    except Exception as e:
        logging.error(f"Error generating Response for instruction '{instruction}': {e}")
        return {
            'reasoning_of_metrics_and_correctness': None,
            'language_quality': None,
            'overall_correctness': None,
            'instruction_following': None,
            'constrains_adherence': None,
            'logical_consistency': None,
            'final_solution_correctness': None,
            'overall_score': None
        }

def process_row(row):
    instruction = row.get('Instruction'.strip())
    system = row.get('System'.strip())
    response = row.get('Chosen'.strip())
    model_generated_output = row.get('model_generated_output'.strip())
    
    if not instruction or not model_generated_output:
        logging.warning(f"Missing 'Instruction' or 'System' in row: {row}")
        return {
            'reasoning_of_metrics_and_correctness': 'Error: Missing data',
            'language_quality': 'Error: Missing data',
            'overall_correctness': 'Error: Missing data',
            'instruction_following': 'Error: Missing data',
            'constrains_adherence': 'Error: Missing data',
            'logical_consistency': 'Error: Missing data',
            'final_solution_correctness': 'Error: Missing data',
            'overall_score': 'Error: Missing data'
        }
    else:
        return generate_model_responses(system, instruction, response, model_generated_output)
    
def process_row_reasoning(row):
    instruction = row.get('Instruction'.strip())
    system = row.get('System'.strip())
    response = row.get('Chosen'.strip())
    model_generated_output = row.get('model_generated_output'.strip())
    
    if not instruction or not model_generated_output:
        logging.warning(f"Missing 'Instruction' or 'System' in row: {row}")
        return {
            'reasoning_of_metrics_and_correctness': 'Error: Missing data',
            'language_quality': 'Error: Missing data',
            'overall_correctness': 'Error: Missing data',
            'instruction_following': 'Error: Missing data',
            'constrains_adherence': 'Error: Missing data',
            'logical_consistency': 'Error: Missing data',
            'final_solution_correctness': 'Error: Missing data',
            'overall_score': 'Error: Missing data'
        }
    else:
        return generate_model_responses_reasoning(system, instruction, response, model_generated_output)

def process_csv_and_generate_output(input_csv_path, output_csv_path):
    try:
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile, delimiter=';')
            fieldnames = [name.strip() for name in reader.fieldnames]
            
            if 'Instruction' not in fieldnames or 'System' not in fieldnames:
                logging.error(f"Input CSV must contain 'Instruction' and 'System' columns. Found columns: {fieldnames}")
                return
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames + ['reasoning_of_metrics_and_correctness', 'language_quality', 'overall_correctness', 'instruction_following', 'overall_score'], delimiter=';')
            writer.writeheader()
            
            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = {executor.submit(process_row, row): row for row in reader}
                
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                        row.update(result)
                    except Exception as e:
                        logging.error(f"Error processing row: {e}")
                        row.update({
                            'reasoning_of_metrics_and_correctness': 'Error: Processing failed',
                            'language_quality': 'Error: Processing failed',
                            'overall_correctness': 'Error: Processing failed',
                            'instruction_following': 'Error: Processing failed',
                            'overall_score': 'Error: Processing failed'
                        })
                    
                    writer.writerow(row)
                    logging.info(f"Processed row with Instruction: {row.get('Instruction', 'N/A')}")
                
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV: {e}")

def process_reasoning_csv_and_generate_output(input_csv_path, output_csv_path):
    try:
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile, delimiter=';')
            fieldnames = [name.strip() for name in reader.fieldnames]
            
            if 'Instruction' not in fieldnames or 'System' not in fieldnames:
                logging.error(f"Input CSV must contain 'Instruction' and 'System' columns. Found columns: {fieldnames}")
                return
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames + ['reasoning_of_metrics_and_correctness', 'language_quality', 'overall_correctness', 'instruction_following', 'constrains_adherence', 'logical_consistency', 'final_solution_correctness', 'overall_score'], delimiter=';')
            writer.writeheader()
            
            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = {executor.submit(process_row_reasoning, row): row for row in reader}
                
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                        row.update(result)
                    except Exception as e:
                        logging.error(f"Error processing row: {e}")
                        row.update({
                            'reasoning_of_metrics_and_correctness': 'Error: Processing failed',
                            'language_quality': 'Error: Processing failed',
                            'overall_correctness': 'Error: Processing failed',
                            'instruction_following': 'Error: Processing failed',
                            'overall_score': 'Error: Processing failed'
                        })
                    
                    writer.writerow(row)
                    logging.info(f"Processed row with Instruction: {row.get('Instruction', 'N/A')}")
                
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred while processing the CSV: {e}")

if __name__ == "__main__":
    input_dir = 'evaluated/extracted'
    output_dir = 'evaluated/judged'
    
    os.makedirs(output_dir, exist_ok=True)
    
    for input_csv_path in glob.glob(os.path.join(input_dir, '*.csv')):
        #Check if the file is the one to be skipped
        if os.path.basename(input_csv_path) == 'extracted_evaluated_hard-reasoning-de_test.csv':
            output_csv_filename = os.path.basename(input_csv_path).replace('evaluated_', 'judged_evaluated_')
            output_csv_path = os.path.join(output_dir, output_csv_filename)
            process_reasoning_csv_and_generate_output(input_csv_path, output_csv_path)
            print(f"Processed {input_csv_path} and saved to {output_csv_path}")
            continue

        if os.path.basename(input_csv_path) == 'extracted_evaluated_hard-reasoning-en_test.csv':
            output_csv_filename = os.path.basename(input_csv_path).replace('evaluated_', 'judged_evaluated_')
            output_csv_path = os.path.join(output_dir, output_csv_filename)
            process_reasoning_csv_and_generate_output(input_csv_path, output_csv_path)
            print(f"Processed {input_csv_path} and saved to {output_csv_path}")
            continue
        
        output_csv_filename = os.path.basename(input_csv_path).replace('evaluated_', 'judged_evaluated_')
        output_csv_path = os.path.join(output_dir, output_csv_filename)
        
        process_csv_and_generate_output(input_csv_path, output_csv_path)
        
        print(f"Processed {input_csv_path} and saved to {output_csv_path}")
