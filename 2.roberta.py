import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
X = data['sentence'].tolist()
y = pd.factorize(data['label'])[0]  # Convert labels to numerical format
label_to_index = {label: i for i, label in enumerate(data['label'].unique())}
index_to_label = {i: label for label, i in label_to_index.items()}

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Prepare datasets and loaders
train_dataset = IntentDataset(X_train, y_train, tokenizer)
val_dataset = IntentDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load pre-trained BERT model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_to_index))
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training and Evaluation Loop
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def evaluate_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset), predictions, true_labels

# Train the model
EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc, val_preds, val_labels = evaluate_epoch(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Save Model
model.save_pretrained('./roberta_intent_model')
tokenizer.save_pretrained('./roberta_intent_model')
