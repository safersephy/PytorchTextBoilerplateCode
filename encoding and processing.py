#encoding in one-hot encoding
import torch 
vocab = ['cat', 'dog', 'rabbit']  
vocab_size = len(vocab)  
one_hot_vectors = torch.eye(vocab_size)  
one_hot_dict = {word: one_hot_vectors[i] for i, word in enumerate(vocab)}  
print(one_hot_dict)

#encoding with count vectorizer
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer()  
corpus = ['This is the first document.', 'This document is the second document.',  
'And this is the third one.', 'Is this the first document?']  
X = vectorizer.fit_transform(corpus)  
print(X.toarray())  
print(vectorizer.get_feature_names_out())

#encoding with tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer()  
corpus = ['This is the first document.','This document is the second document.', 
'And this is the third one.','Is this the first document?']  
X = vectorizer.fit_transform(corpus)  
print(X.toarray())  
print(vectorizer.get_feature_names_out())

#Implementing a Dataset and DataLoader
# Import libraries
from torch.utils.data import Dataset, DataLoader

# Create a class
class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return self.text[idx]

dataset = TextDataset(encoded_text)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#helper functions
def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = tokenizer(sentence)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        freq_dist = FreqDist(tokens)
        threshold = 2
        tokens = [token for token in tokens if freq_dist[token] > threshold]
        processed_sentences.append(' '.join(tokens))
    return processed_sentences


def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    encoded_sentences = X.toarray()
    return encoded_sentences, vectorizer

def extract_sentences(data):
    sentences = re.findall(r'[A-Z][^.!?]*[.!?]', data)
    return sentences

#processing pipeline
def text_processing_pipeline(text):
    tokens = preprocess_sentences(text)
    encoded_sentences, vectorizer = encode_sentences(tokens)
    dataset = TextDataset(encoded_sentences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer

text_data = "This is the first text data. And here is another one."
sentences = extract_sentences(text_data)
dataloaders, vectorizer = [text_processing_pipeline(text) for text in sentences]
print(next(iter(dataloader))[0, :10])
