# pip install torch  -> If you don't have an NVIDIA GPU or don't need CUDA
import torch  # Used for tensor operations and building neural networks
import torch.nn as nn # Provides modules and classes for building neural networks
import torch.optim as optim # Provides optimization algorithms for training neural networks

#=========================================================
# Define the Multilayer Perceptron (MLP) model
# =========================================================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # super() is used to access methods of the superclass(parent class) within a subclass.
        # it calls the `__init__` method of parent class of MLP, which is `nn.Module`. It initializes the inherited part of `nn.Module` within the MLP object.
        super(MLP, self).__init__()

        # Create the first fully connected layer: z = W * x + b
        # x -> input vector of size
        # w -> weight matrix of size
        # b -> bias vector
        # Randomly initialize the w and b parameters, transforming inputs of size `input_size` to outputs of size `hidden_size`
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Rectified Linear Unit(ReLU) -> ReLU(x) = max(0, x)
        # non-linear activation function
        # 1 for x > 0  and 0 for x <= 0
        self.relu = nn.ReLU()

        # creates the final dense layer that takes the hidden_size values from the hidden layer, applies z = W·a₁ + b and produces output_size outputs
        # in the XOR case, 1 value that will pass through the Sigmoid to become the final probability.
        self.fc2 = nn.Linear(hidden_size, output_size)

        # o(z) = 1 / (1 + exp(-z)) -> Sigmoid function
        # (0,1) -> It squashes the input value to a range between 0 and 1, making it suitable for binary classification tasks.
        # o(-∞) = 0
        # o(0) = 0.5
        # o(+∞) = 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """"
            x -> input tensor of shape (batch_size, input_size)
            Forward pass through the network:
            x -> fc1 -> ReLU -> fc2 -> Sigmoid -> output
        """
        # Hidden layer
        # self.fc1(x) computes: z¹ = W¹*x + b¹
        z1 = self.fc1(x)

        # Apply ReLU: a¹ = max(0, z¹)
        a1 = self.relu(z1)

        # Output layer
        # self.fc2(a1) computes: z² = W²*a¹ + b²
        z2 = self.fc2(a1)

        # Apply Sigmoid: o = 1 / (1 + exp(-z²))
        output = self.sigmoid(z2)
        return output

# =================================================
# Learning XOR
# =================================================

# Dataset for the XOR problem
# x_inputs -> inputs -> it is a 2D tensor of shape (4, 2) representing the four possible input combinations for the XOR function.
#        feature 1   feature 2
# sample 0 →  [0,         0]      → XOR(0,0) = 0
# sample 1 →  [0,         1]      → XOR(0,1) = 1
# sample 2 →  [1,         0]      → XOR(1,0) = 1
# sample 3 →  [1,         1]      → XOR(1,1) = 0

# y_targets -> expected outputs
# [0]   ← desired output for (0,0)
# [1]   ← desired output for (0,1)
# [1]   ← desired output for (1,0)
# [0]   ← desired output for (1,1)

# .float32 -> Convert the tensor to a 32-bit floating-point format, which is commonly used for neural network computations.
x_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Create the MLP model
# 2 input features -> 4 hidden neurons -> 1 output neuron

# When MLP() is called, Python executes the `__init__` method of class
# def __init__(self, input_size, hidden_size, output_size):
#    super().__init__()                                # initializes nn.Module
#    self.fc1 = nn.Linear(2, 4)                        # creates W1 (4×2) and b1 (4,)
#    self.relu = nn.ReLU()                             # creates the ReLU
#    self.fc2 = nn.Linear(4, 1)                        # creates W2 (1×4) and b2 (1,)
#    self.sigmoid = nn.Sigmoid()                       # creates the Sigmoid
model = MLP(input_size=2, hidden_size=4, output_size=1)

# The loss function will measure how wrong the network's predictions are.
# BCELoss (Binary Cross Entropy Loss) -> The default loss function for binary classification (0 or 1), such as XOR.

# 4. Why use BCELoss instead of MSE?
# You could technically use nn.MSELoss() (mean squared error), but BCELoss is preferred because:
# It pairs with Sigmoid — the derivative of BCELoss with Sigmoid simplifies to ŷ - y, avoiding gradient saturation.
# Faster convergence — it penalizes confident errors more effectively.
# Probabilistic interpretation — it represents the negative log-likelihood of a Bernoulli distribution.
criterion = nn.BCELoss()

# This line creates the optimizer that will update the network's weights(w) during training. It is the optimizer that, at each epoch, takes the gradients calculated by `loss.backward()` and adjusts w1, b1, w2, b2 ... in the direction that reduces the error.
# model.parameters() returns an iterator over all the parameters (weights and biases) of the model that need to be optimized.
    # [fc1.weight, fc1.bias, fc2.weight, fc2.bias]
# lr = 0.1 -> learning rate

# Why Adam and not SGD?
# SGD (Stochastic Gradient Descent) is a simple and widely used optimization algorithm, but it has some limitations.
# Adam (Adaptive Moment Estimation) is an advanced optimization algorithm that combines the benefits of two other extensions of SGD: AdaGrad and RMSProp. It computes adaptive learning rates for each parameter by keeping track of both the first moment (mean) and the second moment (uncentered variance) of the gradients. This allows Adam to converge faster and handle sparse gradients better than standard SGD.
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ================================================
# Training the model
# ================================================

epochs = 10000

for epoch in range(epochs):

    # Forward pass: compute predicted y by passing x to the model.
    # model is an object of a class that inherits from nn.Module. nn.Module defines the special method __call__, which:
        # Runs some internal PyTorch hooks
        # Calls the forward(X) of your MLP class
        # Returns the result

    # y_pred = model(X)
    # is equivalent to:
    # y_pred = model.forward(X)
    # But always use model(x), never model.forward(x) directly. The __call__ handles important things behind the scenes(hooks, train/eval, etc) that pure forward doesn't.
    y_pred = model(x_inputs)

    # This line calculates the error between what the network predicted(y_pred) and what was expected(y_targets)

    # Loss decreasing → the network is learning ✅
    # Loss constant → the network is not learning (learning rate too low, unsuitable architecture, incorrect data) ❌
    # Loss fluctuating significantly → learning rate too high ⚠️
    # Loss → NaN → something exploded (learning rate too high, log(0), etc.) ❌
    loss = criterion(y_pred, y_targets)

    # Zero out accumulated gradients
    # PyTorch accumulates gradients by default -> PyTorch doesn't replace old gradients. It adds new ones to the existing ones.
    # Epoch 1:
    # loss.backward()   # grad = g1

    # Epoch 2 (without zero_grad):
    # loss.backward()   # grad = g1 + g2   ← accumulated!

    # Epoch 3 (without zero_grad):
    # loss.backward()   # grad = g1 + g2 + g3   ← accumulated further!

    # set_to_none=True (recommended) Fills with None  -> In PyTorch 2.0 or higher, the zero_grad() method already uses set_to_none=True by default.
    # set_to_none=False (old default) Fills with zeros
    optimizer.zero_grad(set_to_none=True)

    # It performs backpropagation, it traverses the computational graph from back to front and calculates the DERIVATIVE of the loss with respect to all the weights(w) and bias(b) of the network.
    loss.backward()

    # Update weights using gradient descent
    # optimizer.step() does: W = W - lr * d(loss)/d(W)

    # What would happen if you did NOT call step()?
        # The gradients are calculated and stored, but never applied. The weights remain the same forever. The network never learns.
    optimizer.step()

    # Prints progress every 1000 epochs
    if(epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}')

# ==================================================================
# Testing the trained model
# ==================================================================
print("\nPredictions after training:")
with torch.no_grad():
    y_pred = model(x_inputs)
    for i in range(len(x_inputs)):
        print(f'Input: {x_inputs[i].numpy()}, Predicted: {y_pred[i].item():.8f}, Target: {y_targets[i].item()}')