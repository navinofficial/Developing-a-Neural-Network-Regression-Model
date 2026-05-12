# Developing a Neural Network Regression Model
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1100" height="699" alt="image" src="https://github.com/user-attachments/assets/5e8bddd0-cc0a-4aa2-911b-71a4351750b7" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Navinkumar V

### Register Number:212223230141

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/Deep learning - Sheet1 - Deep learning - Sheet1.csv')
X = dataset1[['input']].values
y = dataset1[['output']].values
dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10) # Corrected: defined fc2
        self.fc3 = nn.Linear(10,1) # Corrected: renamed to fc3 and proper size
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
        x = self.relu(self.fc1(x)) # Corrected: called as self.fc1(x) and removed extra self from relu
        x = self.relu(self.fc2(x)) # Corrected: called as self.fc2(x) and removed extra self from relu
        x = self.fc3(x)
        return x

lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(lig.parameters())

 def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
     for epoch in range (epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
         print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(lig.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="151" height="194" alt="image" src="https://github.com/user-attachments/assets/6e1f02dc-5683-477e-98a7-c4be13cc648d" />


### OUTPUT
<img width="288" height="182" alt="image" src="https://github.com/user-attachments/assets/39a5d5b4-c637-4221-b7c3-dcc655c9d9c3" />

### Training Loss Vs Iteration Plot
<img width="583" height="451" alt="image" src="https://github.com/user-attachments/assets/7d58d325-afe5-4b2e-83f3-26aaac62d276" />

### New Sample Data Prediction
<img width="250" height="24" alt="image" src="https://github.com/user-attachments/assets/bde21c5e-ea45-4c24-9438-262f488c8cb6" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
