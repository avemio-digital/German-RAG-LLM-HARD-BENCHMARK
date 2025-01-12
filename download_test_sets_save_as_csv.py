import json
import huggingface_hub
from datasets import load_dataset

HF_TOKEN = "hf_[...]"
hf_repo = "avemio/GRAG-HARD-LLM-BENCHMARK"
config_names = [
    'summarize-meeting-attendee-topic', 
    'summarize-meeting-topic', 
    'hard-qa-with-multiple-references', 
    'hard-reasoning-de', 
    'hard-reasoning-en', 
]

# Authenticate with Hugging Face Hub
huggingface_hub.login(HF_TOKEN)

# Iterate over all configurations
for config_name in config_names:
    # Load the dataset with the specified configuration
    dataset = load_dataset(hf_repo, config_name)
    
    # Iterate over all splits
    for split_name in dataset:
        split = dataset[split_name]
        # Check if the split is a test split
        if 'test' in split_name:
            # Check if the required columns are present
            if all(col in split.column_names for col in ["System", "Instruction", "Chosen"]):
                # Save the split as a CSV file
                csv_filename = f"{config_name}_{split_name}.csv"
                split.to_csv(csv_filename, sep=';', index=False, encoding='utf-8')
                print(f"Saved {csv_filename}")