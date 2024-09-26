# Deep learning workshop

### Types of AI

```mermaid
graph LR
    A[ANI - Artificial Narrow Intelligence] -->|Execute specific focused tasks| B[ Limited Functionality]
    C[AGI - Artificial General Intelligence] -->|Perform broad tasks, reason and improve capabilities| D[Comparable to Humans]
    E[ASI - Artificial Super Intelligence] -->|Demonstrate intelligence beyond human capabilities| F[Superior to Humans]
```

### ANI Evolution

- Engineering making programs and intelligent machines [1990 - 1970]
- Ability to learn without being programmed [1980 - 2006]
- Learning based on Deep neural networks [1980 - 2020]

### ML

Machine learning is a field which gives computer the ability to learn without being explicitly programmed.

### ML Types:

```mermaid
graph LR
    A[Machine learning] --> b[Supervised learning]
    A --> c[Unsupervised learning]
    A --> D[Reinforcement Learning]
```

#### Unsupervised learning:

- Clustering, dimensionality reduction

#### Supervised learning:

- Classification, Regression

#### Reinforcement learning:

- Game AI, learning tasks, Real time decisions, Skills acquisition

> NOTE

```
Clustering – process of grouping data, it’s a technique.
Dimensionality reduction.

```

### Traditional programming vs supervised learning

#### traditional programming

```mermaid
graph LR
    A[Input Data] --> B[Computer Program]
    B --> C[Output]
    D[Rules] -->  B
```

#### Supervised learning

```mermaid
graph LR
    A[Input Data] --> B[Computer Program]
    B --> C[Output]
    D[Computer itself produces rules] --> B
```

### Types of data

1.  Structured Data
    - Relational data, a table, follows a structure.
2.  Unstructured Data
    - Scrambelled images, videos etc...
3.  Semi structured Data
    - Html, XML, etc..

### Process of ML Applications

```mermaid
graph LR
    A[Step1: Get Data] --> B[Step2: Preprocess]
    B --> C[Step3: Train Model]
    C --> D[Step4: Test model]
    D --> E[Step5: Improve]
    E --> F[Step6: Deploy]
```

### Classification of KNN

example:

```mermaid
graph LR
a[input <br/> features such as height, weight, size]
b[KNN <br/> algorithm]
c[prediction]
a --> b --> c
```

##### ![alt text](image-2.png)

#### _steps involved in KNN_

```mermaid
graph TD
a[load data and initialize the value of K]
b[iterating from 1 to total number of training data]
c[calculate  the distance between the test data and the training data <br/> euclician distance]
d[sort the caluclated distance in ascending order]
e[get top k rows from the sorted array]
f[get frequent class of rows]
g[return the predicted class]

a --> b --> c --> d --> e --> f --> g
```

### Decision Tree classifier

- Its a type of supervised learning algo. that is mostly used for classification problems.
- it works on both categorical and continuous dependent variables
- In this algo we split hte population into two types

#### Example of iris flower dataset:

```mermaid
graph TB

a[petalwidth] --> iris-setosa-lessthan_0.6
a-->b[petalwidth greater_than_0.6]
b --> pl[petal-length] --> c[petalwidth]
pl --> Iris-versicolor-46
c --> Iris-versicolor
c --> Iris-virginica-3
b --> Isis-virginica-1

```

### performance measures

- A confusion matrix is a table used to describe the performance f=of the classification model or `("classifier")` on a set of test data for which the true values are know.

![alt text](image-3.png)

1. Performance meausres

```
Accuracy = No. of samples predicted correctly / total number of samples
```

2. Precision

```
precision = TP / (TP + FP)
```

3. Recall

```
Recall = TP / (TP + FN)
```

4. Specificity

```
Specifity = TN / (TN + FP)
```

5. F1 SCORE

```
F1 score = 2 * (precision * recall) / (precision + recall)
```

#### Overfitting and underfitting

![alt text](image-4.png)

   <hr/>

### Neural Newtwork

> An artificial neural network(ANN) is a computational model that is inspired by the way biological neural networks in human brain process information.
> <br/><br/>It tries to mimic the behaviour of human brain.

#### Basics of ANN

- Neural networks are typically organised in `layers`.
- Layers are made up of number of interconnected `nodes` which contain `activation function`.
- The inout layer communicates with the external enviorment that presents a pattern to neural network.

> 1. In ANN the input layer consists of numebr of neurons which are equal to the numebr of neurons.<br/><br/>
> 2. In the output layer the number of neurons is equal to the number of classes in the classification problem.

_Lets consider the iris dataset for eg:_

- Input layer have `4` neurons.
- Output layer have `3` layers.

### Tyeps of neural network:

1. **Feed forward networks**

- Single layer
- Multi layer

2. **Feedback networks**
3. **Recurrent neural networks**
4. **Convolutional neural network**

![alt text](image-5.png)

### Recurrent networks:

- they are designed to recognize patterns in sequences of data.

### Convolutional neural network:

- it deals with image data. it shares the weight among neurons using convolutional network.
- it is used in image classification, object detection, image segmentation.

### Actiation function:

- Activation function also called as transfer funciton is used to map input nodes to output nodes in certain fashion.
- Identify linear activation function
- F(x) = X

**1. Binary step function**
$$f(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}$$

**2. Logistic or sigmoid function**
$$f(x) = \frac{1}{1 + e^{-x}}$$

**3. Tanh**
$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**4. ReLU (Rectified Linear Unit)**
$$f(x) = max(0, x)$$

**5. Leaky ReLU**
f(x) = max(ax, x), where x is the input to the neuron, and a is a small constant, typically set to a value like 0.01. This means that when the input x is negative, the output will be ax instead of 0, allowing a small amount of the input to pass through.

**6. Softmax**
f(x_i) = e^(x_i) / ∑(j=1 to n) e^(x_j) where x_i is the i^th element of the input vector, and n is the total number of elements in the input vector.

### How neural networks work

```mermaid
graph LR

a[x1]
b[x2]
c[1]
d[∑]
i[input]

i --> a
i --> b
i --> c

a --> d
b --> d
c --> d

ac[activation function]

d --> ac

po[ Ŷ Predicted Output]

ac --> po

```

Bias = `Ŷ = (b+∑(i=1 to n) x^i *w^i)`

`epoch` describes about number of times the model is trained on the training data.

### preception working

| x1  | x2  | y   |
| --- | --- | --- |
| 158 | 58  | 1   |
| 158 | 59  | 1   |
| 160 | 64  | 0   |
| 163 | 64  | 0   |
| 165 | 61  | 0   |

sample calculation:

```
w1 = 0.4
w2 = 0.5
b = 0.1
θ = 96

yin1 = 153 * 0.4 + 58*0.5 + 0.1
     = 92.3

```

### Perception learning algorithm

P - ipnut with label 1
N - ipnut with label 0
initialize `w` randomly

```mermaid
graph TD

a[While !convergence do]
b[pick random x ∈ P U N]
e[end]
a --> b
b --> c
b --> d
e --> a
c[if x ∈ P and w.x < 0 then] --> s1[w = w + x] --> e

d[if x ∈ N and w.x >= 0 then] --> s2[w = w - x] --> e


```

<hr/>

### ML vs DL

#### Machine learning

```mermaid
graph LR

Input --> Feature_extraction --> Classification --> Output
```

#### Deep learning

```mermaid
graph LR
Input --> a[Feature extraction + Classification] --> Output
```
