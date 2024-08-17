
#BERT model for sequence classification
texts = ["I love this!", "This is terrible.", "Amazing experience!", "Not my cup of tea."]
labels = [1, 0, 1, 0]

import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
inputs["labels"] = torch.tensor(labels)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

model.train()
for epoch in range(1):
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

text = "I had an awesome day!"

input_eval = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs_eval = model(**input_eval)

predictions = torch.nn.functional.softmax(outputs_eval.logits, dim=-1)
predicted_label = 'positive' if torch.argmax(predictions) > 0 else 'negative'

print(f"Text: {text}\nSentiment: {predicted_label}")

#Transformers for classification
sentences = ["I love this product", "This is terrible", "Could be better", "This is the best"]
labels = [1, 0, 0, 1]

train_sentences = sentences[:3]
train_labels = labels[:3]

test_sentences = sentences[3:]
test_labels = labels[3:]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads),
            num_layers=num_layers)
        self.fc = nn.Linear(embed_size, 2)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

model = TransformerEncoder(embed_size=512, heads=8, num_layers=3, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#training
for epoch in range(5):
    for sentence, label in zip(train_sentences, train_labels):
        tokens = sentence.split()
        data = torch.stack([token_embeddings[token] for token in tokens], dim=1)
        output = model(data)
        loss = criterion(output, torch.tensor([label]))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item()}")

#predict 
def predict(sentence):
    model.eval()
    with torch.no_grad():
        tokens = sentence.split()
        data = torch.stack([token_embeddings.get(token, torch.rand((1, 512))) for token in tokens], dim=1)
        output = model(data)
        predicted = torch.argmax(output, dim=1)
    return "Positive" if predicted.item() == 1 else "Negative"

sample_sentence = "This product can be better"
print(f"'{sample_sentence}' is {predict(sample_sentence)}")


#RNN with attention

#preprocessing
data = ["the cat sat on the mat", ...]
vocab = set(' '.join(data).split())
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

pairs = [sentence.split() for sentence in data]
input_data = [[word_to_ix[word] for word in sentence[:-1]] for sentence in pairs]
target_data = [word_to_ix[sentence[-1]] for sentence in pairs]

inputs = [torch.tensor(seq, dtype=torch.long) for seq in input_data]
targets = torch.tensor(target_data, dtype=torch.long)

#model definition
embedding_dim = 10
hidden_dim = 16

class RNNWithAttentionModel(nn.Module):
    def __init__(self):
        super(RNNWithAttentionModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        attn_weights = torch.nn.functional.softmax(self.attention(out).squeeze(2), dim=1)
        context = torch.sum(attn_weights.unsqueeze(2) * out, dim=1)
        out = self.fc(context)
        return out

    def pad_sequences(batch):
        max_len = max([len(seq) for seq in batch])
        return torch.stack([torch.cat([seq, torch.zeros(max_len - len(seq)).long()]) for seq in batch])

criterion = nn.CrossEntropyLoss()
attention_model = RNNWithAttentionModel()
optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.01)

for epoch in range(300):
    attention_model.train()
    optimizer.zero_grad()
    
    padded_inputs = pad_sequences(inputs)
    outputs = attention_model(padded_inputs)
    
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()



for input_seq, target in zip(input_data, target_data):
    input_test = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
    
    attention_model.eval()
    attention_output = attention_model(input_test)
    attention_prediction = ix_to_word[torch.argmax(attention_output).item()]
    
    print(f"Input: {' '.join([ix_to_word[ix] for ix in input_seq])}")
    print(f"Target: {ix_to_word[target]}")
    print(f"RNN with Attention prediction: {attention_prediction}")
