import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class LengthPredictor:
    def __init__(self, model_path, max_length=512, bucket_size=128, num_buckets=8):
        self.max_length = max_length
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

        self.device = torch.device('cpu')
        self.model.to(self.device)
    
    def predict_length(self, prompt):
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        if predicted_class == 0:
            return self.bucket_size
        else:
            return (predicted_class + 1) * self.bucket_size

if __name__ == "__main__":
    model_path = '/home/huafeng/SInfer/sinfer/predictor/distilbert-based-uncased-finetuned'
    predictor = LengthPredictor(model_path)

    prompt = "Identify the odd one out. Twitter, Instagram, Telegram"
    predicted_length = predictor.predict_length(prompt)
    print(f"Predicted length: {predicted_length}")