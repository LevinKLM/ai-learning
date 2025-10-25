## D2L
**Chapters:** 1.1–1.2

### Preface
  1. Workspace setup
  * Miniconda: used to manage packages and environments in Python. 
### 1.1 A Motivating Example
  1. Summaries:
  * Problem is different in nature, usually no deterministic algorithm can be applied.
  * Concepts: dataset, model, model family, learning algorithm.
  2. Question:
  * The example used is to build a model that recognize a keyword phrase and output "yes", "no". How about if we don't know the keyword and want to classify the input into different groups of words? Yes, it can be done through unsupervised learning (in later chapters).
### 1.2 Key Components
  1. Summaries:
  * Concepts: supervised learning, data, model, objective functions, optimization algorithms.
### 1.2.1 Data
  1. Summaries:
  * Concepts: example/data/data points/data instances/sample, features/covariates/inputs, label/target, fixed-length vectors, dimensionality, varying-length data.
  * Garbage data can really hurt, can lead to poor performance, unrepresented data, societal prejudices, etc.
### 1.2.2 Models
  1. Summaries:
  * Concepts: model, statistical model, deep learning.
### 1.2.3 Objective functions
  1. Summaries:
  * Concepts: objective functions, loss functions, squared error, surrogate objective, training dataset, test dataset, overfitting.
### 1.2.4 Optimization Algorithms
  1. Summaries:
  * Concepts: gradient descent.
### 1.3 Kinds of Machine Learning Problems
### 1.3.1 Supervised Learning
  1. Regression
     * When labels take on arbitrary numerical values.
     * Goal is to predict as accurately as possible.
     * How many/much? problem
  2. Classification
     * Which one? problem

## Others
  1. A problem shown at work that the analysis of a crash dump of a minified program is difficult and time consuming.
  * I was thinking more about how to categorize the problem, if I can use some AI model to categorize the dumps, then the analysis of the dumps will have more direction (starting with the cluster with highest count).