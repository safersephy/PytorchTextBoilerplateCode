import torch 
import torch.nn as nn  
data = "Hello how are you?" 
chars = list(set(data))   
char_to_ix = {char: i for i, char in enumerate(chars)} 
ix_to_char = {i: char for i, char in enumerate(chars)}  
class RNNModel(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(RNNModel, self).__init__() 
        self.hidden_size = hidden_size  
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)

def forward(self, x):
    h0 = torch.zeros(1, x.size(0), self.hidden_size)
    out, _ = self.rnn(x, h0)
    out = self.fc(out[:, -1, :])
    return out

model = RNNModel(1, 16, 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#preprocessing
inputs = [char_to_ix[ch] for ch in data[:-1]]
targets = [char_to_ix[ch] for ch in data[1:]]

inputs = torch.tensor(inputs, dtype=torch.long).view(-1, 1)
inputs = nn.functional.one_hot(inputs, num_classes=len(chars)).float()

targets = torch.tensor(targets, dtype=torch.long)

#training
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item()}')

#testing
model.eval()
test_input = char_to_ix['h']

test_input = nn.functional.one_hot(torch.tensor(test_input).view(-1, 1), num_classes=len(chars)).float()
predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()

print(f'Test Input: 10, Predicted Output: {model(test_input).item()}')

#GAN
# Embedding reviews
# Convert reviews to tensors

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, seq_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()

optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

#training
num_epochs = 50

for epoch in range(num_epochs):
    for real_data in data:
        real_data = real_data.unsqueeze(0)
        noise = torch.rand((1, seq_length))
        
        disc_real = discriminator(real_data)
        fake_data = generator(noise)
        disc_fake = discriminator(fake_data.detach())
        
        loss_disc = criterion(disc_real, torch.ones_like(disc_real)) + \
                    criterion(disc_fake, torch.zeros_like(disc_fake))
        
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

disc_fake = discriminator(fake_data)
loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))

optimizer_gen.zero_grad()
loss_gen.backward()
optimizer_gen.step()

if (epoch+1) % 10 == 0:
    print(f"Epoch {epoch+1}/{num_epochs}:\tGenerator loss: {loss_gen.item()}\tDiscriminator loss: {loss_disc.item()}")

print("\nReal data: ")
print(data[:5])

print("\nGenerated data: ")
for _ in range(5):
    noise = torch.rand((1, seq_length))
    generated_data = generator(noise)
    print(torch.round(generated_data).detach())
