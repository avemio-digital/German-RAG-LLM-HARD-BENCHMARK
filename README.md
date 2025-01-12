# GRAG-LLM-HARD-BENCHMARK

## Overview

This repository provides a framework for evaluating models compatible with OpenAI endpoints. The evaluation process involves downloading datasets, preparing them, and then running evaluations using the OpenAI API. The results are further processed to extract special metrics and evaluated using a language model as a judge.

### Dependencies

Ensure the following dependencies are installed before running the scripts:

- `openai` Python package

`pip install openai`

## Steps to Evaluate a Model

### 1. Download Evaluation Dataset

First, download the evaluation dataset from Hugging Face using the `python download_test_sets_save_as_csv.py`. The dataset can be found at the following link: [GRAG-LLM-HARD-BENCHMARK](https://huggingface.co/datasets/avemio/GRAG-LLM-HARD-BENCHMARK).

### 2. Prepare Datasets

Prepare the datasets and extract special patterns necessary for evaluation using the `python prepare_datasets.py`

### 3. Evaluate the Model

To evaluate a model compatible with an OpenAI endpoint, use `python evaluate_model.py`. Ensure you have set a valid OpenAI API key. This can be done via an environment variable, an environment file, or directly in the code (not recommended for security reasons).

### 4. Extract Special Metrics

Navigate to the `evaluated` directory with `cd evaluated` and execute the `python extract_special_metrics.py` script to process the evaluation results.

### 5. Evaluate by Judge

Return to the root directory with `cd ..` and execute `python evaluate_by_judge.py` to evaluate your model using a language model as a judge.

### 6. Generate weighted Files & Plots

Open the `GRAG-LLM-HARD-BENCHMARK.ipynb` and execute all cells.

## Setting Up OpenAI API Key

To evaluate a model from OpenAI, you must set a valid OpenAI API key. This can be done in one of the following ways:

- Set an environment variable: `export OPENAI_API_KEY='your-api-key'`
- Use an environment file: Create a `.env` file with the line `OPENAI_API_KEY=your-api-key`
- Directly in the code (not recommended): Replace the placeholder in the code with your API key.

## Code References

The `evaluate_model.py` script is responsible for generating model responses and processing CSV files. Key functions include:
- `generate_model_responses`: Handles communication with the OpenAI API.

``` 
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
```


## Additional Notes

- Ensure that the input CSV files are placed in the `prepared` directory and the output will be saved in the `evaluated` directory.
- The script logs important information and errors, which can be useful for debugging and tracking the evaluation process.
