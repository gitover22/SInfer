import json
from transformers import DistilBertTokenizer

MAX_OUTPUT_LENGTH = 1024
NUM_BUCKETS = 8

bucket_size = MAX_OUTPUT_LENGTH // NUM_BUCKETS

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-based-uncased')

def get_bucket(token_length):
    if token_length >= MAX_OUTPUT_LENGTH:
        return NUM_BUCKETS - 1
    return token_length // bucket_size

def transform_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    transformed_data = []
    
    for item in data:
        prompt = f"{item['instruction']} {item['input']}"
        output_ids = tokenizer.encode(item['output'])
        output_ids_length = len(output_ids)
        bucket = get_bucket(output_ids_length)
        
        transformed_data.append({
            "prompt": prompt,
            "output_length_bucket": bucket
        })
    
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=4)

input_file = 'predictor/alpaca_data.json'
output_file = 'predictor/alpaca_data_postprocess.json'  
transform_data(input_file, output_file)