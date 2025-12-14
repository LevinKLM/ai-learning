## D2L
**Chapters:** 1.6-2.3.7

### 1.6 Success Stories
  * AI applications: handwritten recognition, search/recommendation/ranking, AI assistants, speech recognition, object identification in image, games (Chess/Go/Poker), self-driving

### 1.7 The Essence of Deep Learning
  * No more feature engineering, it is replaced by automatically tuned filters that yield superior accuracy
  * Eliminated many of the boundaries between computer vision, speech recognition, NLP, etc.
  * Nonparametric model
  * Acceptance of suboptimal solutions

### 1.8 Summary
### 1.9 Exerciess
  * Algorithm selection must simultaneously consider the scale and structure of the data and the budget and efficiency of the available computational resources. Larger datasets allow for more complex models, and greater computational power enables larger and more expressive architectures. The most appropriate algorithm is ultimately the one that balances data characteristics, model capacity, and computational constraints.
### 2.1 Data Manipulation
  * torch and tensor, basic operations
  * Broadcasting: each dimension must match or be 1, broadcasting from right to left (from element to higher dimentions)
  * In place operation: use [:]
### 2.2 Data Preprocessing
  * Read CSV data with pandas
  * Handle missing values
  * Convert to tensor
### 2.3 Linear Algebra
  * Scalar, vector, matrix, Tensor
  * Tensor Arithmetic
  * Reduction
  * Non-Reduction Sum


## Others
* Skipped week 20251102
* Confused between the index and dimension in einsum, conclusion:
  * A dimension is part of the tensor’s shape.
  * An index (i, j, k, …) is a variable that iterates over a dimension.
  * Identical indices = aligned dimensions (they correspond).
  * Repeated indices = summation (contraction) and the index disappears.
  * An index is never the dimension itself.