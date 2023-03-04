import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class for binary classification data
class BinaryClassificationDataset(Dataset):
    def __init__(self, data):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define a PyTorch Lightning module for the RoBERTa model
class RoBERTaClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        num_labels = self.model.config.num_labels
        self.model.classifier = torch.nn.Linear(self.model.pooler.dense.out_features, num_labels)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids, attention_mask=attention_mask, labels=labels)
        loss = self.loss_function(outputs.logits, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer

print("Loading data...")
# Load the training data from a CSV file
train_data = pd.read_csv('/vol/bitbucket/es1519/NLPClassification_01/roberta_model/DontPatronizeMe/csv_files/dontpatronizeme_pcl_train.csv')

# Create a PyTorch Lightning trainer and train the model
train_dataset = BinaryClassificationDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print("Dataset created")
model = RoBERTaClassifier()
print("Training...")
trainer = pl.Trainer(max_epochs=3, gpus=1)
trainer.fit(model, train_dataloader)

# Save the trained model
model.model.save_pretrained('/vol/bitbucket/es1519/NLPClassification_01/roberta_model/DontPatronizeMe/trained_model/')
