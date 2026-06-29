# pytorch https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np
from kpal.klib import Profile

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# or create from numpy array:
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# new tensors retain properties of old unless explicitly overwritten
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described here.


# need a tutorial that actually does something, not just going through each element
# try https://www.geeksforgeeks.org/start-learning-pytorch-for-beginners/
# Define tensors with requires_grad=True to track computation history

# autograd: automatic calculation of the gradients
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Perform a computation
z = x**2 + y**3
print("Output tensor z:", z)

# Compute gradients of z with respect to x and y
z.backward()
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)

## build simple network with iris dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # we are defining the input layer (fc1) that contains the linear transformation (nn.Linear) to map the input features to the hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer
        #  apply the ReLU activation function (nn.ReLU) to introduce non-linearity in the model. 
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

# Set random seed for reproducibility
torch.manual_seed(42)


# Define the input size, hidden size, and output size of the neural network
input_size = X.shape[1]
hidden_size = 10
output_size = len(iris.target_names)


# Instantiate the neural network
model = SimpleNN(input_size, hidden_size, output_size)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Convert datto PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)


# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
   
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   
    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluate the model
# convert the test dataset from NumPy Arrays into the PyTorch Sensors using the torch.FloatTensor() and torch.LongTensor(). Then, we pass the test input X_test_tensor through the trained model to obtain the output. At last, these are compared with the actual predicted labels y_test_tensor to calculate the accuracy. 
with torch.no_grad(): # why no grad?
    # Disabling gradient calculation is useful for inference, when you are sure
    # that you will not call :meth:`Tensor.backward()`. It will reduce memory
    # consumption for computations that would otherwise have `requires_grad=True`.

    # In this mode, the result of every computation will have
    # `requires_grad=False`, even when the inputs have `requires_grad=True`.
    # There is an exception! All factory functions, or functions that create
    # a new Tensor and take a requires_grad kwarg, will NOT be affected by
    # this mode.

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on the test set: {accuracy:.2f}')



# efficient data handling are crucial while learning PyTorch. So in this section, we will learn about various data handling techniques like Data Loading and Preprocessing. 


### so if i was to make a custom dataset for sequences,
# i could, e.g. make a kmer decomposition of each
# maybe check https://kmer.readthedocs.io/en/stable/method.html
class KmerDataset(Dataset):
    # he Datasets class acts as the interface for custom datasets. 
    # You have to use the ‘len’ and ‘getitem’ methods to create Custom dataset for model building using PyTorch. 
    def __init__(self, data, targets, k):
        self.data = self.count_kmers(data, k)
        self.targets = targets

    def count_kmers(self, sequence, k_size):
        """
        Need to take tensors as Input, and Output Tensor
        Would the label be the count? No, label would be sequence label
        Can you have mixed data types in tensor? e.g. kmer and count
        """
        data = {}
        size = len(sequence)
        for i in range(size - k_size + 1):
            kmer = sequence[i: i + k_size]
            try:
                data[kmer] += 1
            except KeyError:
                data[kmer] = 1
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    


# Custom Dataset class
class CustomDataset(Dataset):
    # he Datasets class acts as the interface for custom datasets. 
    # You have to use the ‘len’ and ‘getitem’ methods to create Custom dataset for model building using PyTorch. 
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Example data
data = torch.randn(100, 3, 32, 32)  # Example image data
targets = torch.randint(0, 10, (100,))  # Example target labels


# Create custom dataset
custom_dataset = CustomDataset(data, targets)


# Create DataLoader
batch_size = 32
shuffle = True
num_workers = 4
#  DataLoader iterates over the dataset and fetches batches of sample
data_loader = DataLoader(custom_dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)


# Iterate over batches
for batch_idx, (inputs, targets) in enumerate(data_loader):
    print(
        f"Batch {batch_idx+1}: Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

## would model above then be able to work with this dataset?

# Preprocessing of the data means bringing the data into the standard format so that data can be fitted into the model. Here, the two main methods are Transformation and Normalization

import torchvision.transforms as transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),              # Resize images to 256x256
    transforms.RandomCrop(224),          # Randomly crop images to 224x224
    transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
    transforms.ToTensor(),               # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Normalize images
]) 
# conda install pytorch-gpu torchvision torchaudio pytorch-cuda=11.4 -c pytorch -c nvidia
# no pytorch-cuda=11.4
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f 

# Example of applying transformations to image
example_image = transforms.ToPILImage()(
    torch.randn(3, 256, 256))  # Example image tensor
transformed_image = transform(example_image)


print("Transformed image shape:", transformed_image.shape)


# Define custom dataset class by subclassing torch.utils.data.Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)
      
    def __getitem__(self, index):
        # Retrieve and return a sample and its corresponding target based on the given index
        sample = self.data[index]
        target = self.targets[index]
        return sample, target


# Example data and targets
data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
targets = torch.tensor([0, 1, 0, 1])

# Create instance of the custom dataset
custom_dataset = CustomDataset(data, targets)

# Create a data loader to iterate over the dataset in batches
batch_size = 2
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch_idx, (samples, targets) in enumerate(data_loader):
    print(f"Batch {batch_idx}:")
    print("Samples:", samples)
    print("Targets:", targets)

### so if i was to make a custom dataset for sequences,
# i could, e.g. make a kmer decomposition of each
# maybe check https://kmer.readthedocs.io/en/stable/method.html
class KmerDataset(Dataset):
    # he Datasets class acts as the interface for custom datasets. 
    # You have to use the ‘len’ and ‘getitem’ methods to create Custom dataset for model building using PyTorch. 
    def __init__(self, data, targets, k):
        self.data = self.count_kmers(data, k)
        self.targets = targets

    def count_kmers(self, sequence, k_size):
        """
        Need to take tensors as Input, and Output Tensor
        Would the label be the count? No, label would be sequence label
        Can you have mixed data types in tensor? e.g. kmer and count
        """
        data = {}
        size = len(sequence)
        for i in range(size - k_size + 1):
            kmer = sequence[i: i + k_size]
            try:
                data[kmer] += 1
            except KeyError:
                data[kmer] = 1
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    

# potentially
# Unsupervised approach for feature (kmer embeddings) extraction from a provided reference genome. This code is built on the word2vec model by Mikolov et al. 
#https://github.com/Ethan-Loo/kmer2vec_pytorch

# they convert kmers to ints here: https://github.com/Ethan-Loo/kmer2vec_pytorch/blob/master/utils_torch.py
# uses recursive method, presumably in some way guaranteed to produce unique ints for each kmer

# can probably steal a lot of code for use in a supervised learning algorithm
# do we want to include all kmers? or are some more "informative" than others. or is the totality the informative bit?
# can we do something to determine whether some are more informative than others? they would have higher weight- but can you giv eindividual components higher weight?

# i would do set kmer size, not multiple

