<H3>Name:Anikha Pillai</H3>
<H3>Register no. 212225230016</H3>
<H3>Date: 21/11/26</H3>
<H3>Experiment-2 </H3>
## Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.b = 0.0
        self.w = None
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array) -> int:
        return np.where(self.f(x) >= 0, 1, -1)


# Load the Breast Cancer binary classification dataset
df = pd.read_csv("/content/breast_cancer_binary_dataset.csv")
print(df.head())

# Select the target and first three numerical features
y = df["diagnosis"].values
feature_names = ["mean radius", "mean texture", "mean perimeter"]
x = df[feature_names].values

# Convert class names to numeric labels for plotting and training
# Malignant = 0, Benign = 1
y_numeric = np.where(y == "Malignant", 0, 1)

# 3D scatter plot using three features
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_title("Breast Cancer Dataset")
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean texture")
ax.set_zlabel("Mean perimeter")

ax.scatter(
    x[y_numeric == 0, 0], x[y_numeric == 0, 1], x[y_numeric == 0, 2],
    color="red", marker="o", s=8, label="Malignant"
)
ax.scatter(
    x[y_numeric == 1, 0], x[y_numeric == 1, 1], x[y_numeric == 1, 2],
    color="blue", marker="^", s=8, label="Benign"
)

plt.legend(loc="upper left")
plt.show()

# Use the first two features for the 2D plot and Perceptron
x = x[:, 0:2]

plt.figure(figsize=(5, 5))
plt.scatter(
    x[y_numeric == 0, 0], x[y_numeric == 0, 1],
    color="red", marker="o", s=12, label="Malignant"
)
plt.scatter(
    x[y_numeric == 1, 0], x[y_numeric == 1, 1],
    color="blue", marker="x", s=12, label="Benign"
)

plt.xlabel("Mean radius")
plt.ylabel("Mean texture")
plt.title("Breast Cancer Binary Classification")
plt.legend(loc="upper left")
plt.show()

# Perceptron uses labels -1 and +1
# Malignant = +1, Benign = -1
y = np.where(y_numeric == 0, 1, -1)

# Split first, then standardize using training data only
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train, n_iter=10)

y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

# Plot the number of misclassified samples in each epoch
plt.figure(figsize=(5, 4))
plt.plot(
    range(1, len(classifier.misclassified_samples) + 1),
    classifier.misclassified_samples,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Perceptron Learning")
plt.show()
```

# OUTPUT:

 <img width="564" height="564" alt="image" src="https://github.com/user-attachments/assets/564bbba6-cde3-4b9d-9c35-f1b0a6e504a8" />

 <img width="601" height="627" alt="image" src="https://github.com/user-attachments/assets/d0806e1b-b893-4569-ade1-d0c13f0a2aa6" />

 <img width="627" height="574" alt="image" src="https://github.com/user-attachments/assets/805db49b-cc59-4da2-8995-3f7196009b48" />




# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify breast cancer data set.

 
