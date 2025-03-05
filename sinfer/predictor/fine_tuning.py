import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MAX_LENGTH = 512 
BATCH_SIZE = 16 
EPOCHS = 10     
LEARNING_RATE = 2e-5 
NUM_BUCKETS = 8
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-based-uncased')

class OutputLengthDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        label = item['output_length_bucket']
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

with open('predictor/alpaca_data_postprocess.json', 'r') as f:
    data = json.load(f)

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = OutputLengthDataset(train_data, tokenizer, MAX_LENGTH)
val_dataset = OutputLengthDataset(val_data, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = DistilBertForSequenceClassification.from_pretrained(
    'predictor/distilbert-based-uncased',
    num_labels=NUM_BUCKETS 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Training loss: {avg_loss}")

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Validation loss: {avg_loss}, Accuracy: {accuracy}")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, optimizer, device)
    evaluate(model, val_loader, device)

model.save_pretrained('predictor/distilbert-based-uncased-finetuned')
tokenizer.save_pretrained('predictor/distilbert-based-uncased-finetuned')