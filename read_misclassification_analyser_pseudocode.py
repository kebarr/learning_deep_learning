# Step 1: Data Prep
sequence_reads, labels = load_data()
encoded_sequences = encode(sequence_reads)  # Here you could encode them with something like kmer frequencies, or anything, just try and make your incoding as invariant as possible to the initial locus. FCGR will work but will give you an image as opposed to a linear sequence, so we'd need to slightly modify the model, not a problem but yeah.

train_data, val_data, test_data = split_data(encoded_sequences, labels)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) # BATCH_SIZE dictates how many sequence-label pairs the model sees before it runs an update on itself. Go with 32 or 64 initially i expect.

val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Step 2: Define Neural Network using e.g. pytorch (torch.nn)
class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SequenceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize model
input_size = len(encoded_sequences[0])  # Length of input vector (e.g. if kmer frequency encoding, this will be 4**k)
hidden_size = 128  # Example size for the "hidden" layer of neurons in our simple classifier model
model = SequenceClassifier(input_size, hidden_size)

# Step 3: Define Training Parameters
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr is initial learning rate, but Adam tends to find a good learning rate as it runs, i.e. less critical to tune this than it used to be.

# Step 4: Train the Model
for epoch in range(NUM_EPOCHS): # 1 epoch is 1 cycle of the full training data set
    model.train()
    for batch_sequences, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_sequences.float())
        loss = criterion(outputs, batch_labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Validate model
    model.eval()
    with torch.no_grad():
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Step 5: Evaluate the Model
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Step 6: Save the Model
torch.save(model.state_dict(), "sequence_classifier.pth")