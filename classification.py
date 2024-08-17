
#embedding

import torch
from torch import nn

words = ["The", "cat", "sat", "on", "the", "mat"]
word_to_idx = {word: i for i, word in enumerate(words)}
inputs = torch.LongTensor([word_to_idx[w] for w in words])
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)
output = embedding(inputs)
print(output)


#pipelining
def preprocess_sentences(text):
    # Tokenization
    # Stemming
    ...
    # Word to index mapping

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, encoded_sentences):
        self.data = encoded_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def text_processing_pipeline(text):
    tokens = preprocess_sentences(text)
    dataset = TextDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer

text = "Your sample text here."
dataloader, vectorizer = text_processing_pipeline(text)
embedding = nn.Embedding(num_embeddings=10, embedding_dim=50)
for batch in dataloader:
    output = embedding(batch)
    print(output)



#CNN classifier
class SentimentAnalysisCNN(nn.Module):  
    def __init__(self, vocab_size, embed_dim):  
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embed_dim)  
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)  
        self.fc = nn.Linear(embed_dim, 2) 
        
    def forward(self, text): 
        embedded = self.embedding(text).permute(0, 2, 1)  
        conved = F.relu(self.conv(embedded))  
        conved = conved.mean(dim=2)  
        return self.fc(conved)
    
vocab = ["i", "love", "this", "book", "do", "not", "like"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(word_to_idx)
embed_dim = 10

book_samples = [
    ("The story was captivating and kept me hooked until the end.".split(), 1),
    ("I found the characters shallow and the plot predictable.".split(), 0)
]

model = SentimentAnalysisCNN(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#training loop
for epoch in range(10):
    for sentence, label in data:
        model.zero_grad()
        sentence = torch.LongTensor([word_to_idx.get(w, 0) for w in sentence]).unsqueeze(0)
        outputs = model(sentence)
        label = torch.LongTensor([int(label)])
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()


#running the model
for sample in book_samples:
    input_tensor = torch.tensor([word_to_idx[w] for w in sample], dtype=torch.long).unsqueeze(0)
    outputs = model(input_tensor)
    _, predicted_label = torch.max(outputs.data, 1)
    sentiment = "Positive" if predicted_label.item() == 1 else "Negative"
    print(f"Book Review: {' '.join(sample)}")
    print(f"Sentiment: {sentiment}\n")


#LSTM classifier
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

#GRU classifier
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output

#evaluation
from torchmetrics import Accuracy
actual = torch.tensor([0, 1, 1, 0, 1, 0])
predicted = torch.tensor([0, 0, 1, 0, 1, 1])
accuracy = Accuracy(task="binary", num_classes=2)
acc = accuracy(predicted, actual)
print(f"Accuracy: {acc}")

from torchmetrics import Precision, Recall
precision = Precision(task="binary", num_classes=2)
recall = Recall(task="binary", num_classes=2)
prec = precision(predicted, actual)
rec = recall(predicted, actual)
print(f"Precision: {prec}")
print(f"Recall: {rec}")

from torchmetrics import F1Score
f1 = F1Score(task="binary", num_classes=2)
f1_score = f1(predicted, actual)
print(f"F1 Score: {f1_score}")