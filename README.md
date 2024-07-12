## Linear Regression House Price Estimation

This repository contains a C program that implements a simple machine learning algorithm to estimate house prices based on historical data. The program uses a one-shot learning approach to deduce the weights of different house attributes, which can then be used to predict prices for new houses.

## Project Description

This project aims to provide hands-on experience with programming in C, file I/O, dynamic memory allocation, and implementing a moderately complex algorithm in a Unix environment. The primary goal is to predict house prices based on a given set of attributes by calculating weights using a one-shot learning algorithm.

### Algorithm Overview

The algorithm uses the following formula to estimate the price of a house:

\[ y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 \]

Where:
- \( y \) is the estimated price of the house.
- \( x_1, x_2, x_3, x_4 \) are the attributes of the house (e.g., number of bedrooms, bathrooms, square footage, year built).
- \( w_0, w_1, w_2, w_3, w_4 \) are the weights corresponding to each attribute.

The algorithm calculates the weights using the provided training data and then uses these weights to estimate prices for new houses.

### Matrix Representation

- **Training Data (X):** An \( n \times (k + 1) \) matrix where each row represents a house and each column represents an attribute. The first column is filled with 1s to account for the intercept weight \( w_0 \).
- **House Prices (Y):** An \( n \times 1 \) matrix where each row gives the price of a house.
- **Weights (W):** A \( (k + 1) \times 1 \) matrix where each row gives the weight of an attribute.

The relationship between these matrices is given by the equation:

\[ XW = Y \]

To find the weights \( W \), the algorithm uses the pseudo-inverse of the matrix \( X \):

\[ W = (X^TX)^{-1}X^TY \]

### Implementation Steps

1. **Matrix Multiplication:** Multiply matrices as required by the algorithm.
2. **Matrix Transposition:** Transpose the training data matrix \( X \).
3. **Gauss-Jordan Elimination:** Implement Gauss-Jordan elimination to invert the matrix \( X^TX \).

## Usage

The program `estimate` reads the training data and input data from files, computes the weights, and outputs the estimated prices.

### Input File Format

**Training Data File:**
```
train
k (number of attributes)
n (number of houses)
x1 x2 x3 x4 y (attributes and price of each house)
...
```

**Input Data File:**
```
data
k (number of attributes)
m (number of houses)
x1 x2 x3 x4 (attributes of each house)
...
```

### Output

The program prints the estimated prices for each house in the input data, rounded to the nearest integer.

### Example

**Training Data (train.txt):**
```
train
4
7
3.0 1.0 1180.0 1955.0 221900.0
3.0 2.25 2570.0 1951.0 538000.0
2.0 1.0 770.0 1933.0 180000.0
4.0 3.0 1960.0 1965.0 604000.0
3.0 2.0 1680.0 1987.0 510000.0
4.0 4.5 5420.0 2001.0 1230000.0
3.0 2.25 1715.0 1995.0 257500.0
```

**Input Data (data.txt):**
```
data
4
2
3.0 2.5 3560.0 1965.0
2.0 1.0 1160.0 1942.0
```

**Command:**
```
$ ./estimate train.txt data.txt
```

**Output:**
```
737861
203060
```

## Compilation and Execution

1. **Compile the Program:**
   ```sh
   gcc -o estimate estimate.c
   ```

2. **Run the Program:**
   ```sh
   ./estimate train.txt data.txt
   ```

## Requirements

- C compiler (e.g., GCC)
- Unix environment for development and testing

## License

This project is licensed under the MIT License.
