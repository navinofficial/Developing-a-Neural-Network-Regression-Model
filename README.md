# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A Neural Network Regression Model is a deep learning technique used to predict continuous numerical values from input data. In this program, the regression model is implemented using the PyTorch library. The model learns the relationship between input and output values through training and minimizes prediction error using optimization techniques.

The dataset is first loaded using the Pandas library. The input and output values are separated into features (X) and target values (y). The dataset is divided into training and testing data using train_test_split() from Scikit-learn. Data normalization is performed using MinMaxScaler to scale the values between 0 and 1, which improves the training performance of the neural network.

The neural network consists of:

Input Layer – accepts one input feature.
Hidden Layers – two fully connected layers with 8 and 10 neurons.
Activation Function – ReLU (Rectified Linear Unit) introduces non-linearity.
Output Layer – produces a single continuous numerical output.

The model is created by inheriting the nn.Module class in PyTorch. Forward propagation is implemented in the forward() function, where input data passes through hidden layers and activation functions to generate predictions.

The training process uses:

Loss Function: Mean Squared Error (MSELoss), which calculates the difference between predicted and actual values.
Optimizer: RMSprop optimizer, which updates model weights efficiently during backpropagation.
Epochs: The model is trained repeatedly for 2000 iterations to reduce loss.

During training, the loss value is stored and plotted using Matplotlib to visualize model learning performance. After training, the model is evaluated using test data, and predictions are generated for new input values.

## Neural Network Model
<img width="852" height="634" alt="image" src="https://github.com/user-attachments/assets/8d5afcac-cb9d-48d0-abaa-1fe42e2bdda5" />


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

### Name:  Navinkumar V

### Register Number:  212223230141

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('/content/DEEP DOCS - Sheet1.csv')
X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values
dataset1.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# Name:  Navinkumar V
# Register Number: 212223230141
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'Loss': []}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x




lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(lig.parameters(),lr=0.001)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Name: Navinkumar V
# Register Number: 212223230141
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      lig.history['Loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

import matplotlib.pyplot as plt
plt.plot(lig.history['Loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
<img width="298" height="296" alt="image" src="https://github.com/user-attachments/assets/b6ff5016-4603-47c5-97a6-84882855055c" />


### OUTPUT
<img width="375" height="230" alt="image" src="https://github.com/user-attachments/assets/8f307a66-564c-45df-8959-45117317b04c" />
<img width="269" height="42" alt="image" src="https://github.com/user-attachments/assets/08b3a758-f8f9-417b-8eda-436f1e78811d" />






### Training Loss Vs Iteration Plot
<img width="754" height="583" alt="image" src="https://github.com/user-attachments/assets/adab5547-e919-4ba8-9741-5f291551986c" />

### New Sample Data Prediction
<img width="400" height="35" alt="image" src="https://github.com/user-attachments/assets/900bc538-d2a0-4a49-b32e-75fa517e5465" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
