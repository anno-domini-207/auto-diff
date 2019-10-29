# CS207 Final Project: Milestone 1

**Group 15 : Simon Warchol, Kaela Nelson, Qiuyang Yin, Yoon S. Park**

## 1. Introduction

In machine learning, calculating the derivative and gradients of functions is essential. This is because machine learning algorithms are centered around minimizing an objective loss function. Traditionally, scientists have used numerical differentiation methods to compute these derivatives and gradients, which potentially accumulates floating point errors in calculations and penalizes accuracy.

Automatic differentiation is an algorithm that can solve complex derivatives in a way that reduces these compounding floating point errors. The algorithm achieves this by breaking down functions into their elementary components and then calculates and evaluates derivatives at these elementary components. This allows the computer to solve derivatives and gradients more efficiently and precisely. This is a huge contribution to machine learning, as it allows scientists to achieve results with more precision.

## 2. Background

In automatic differentiation, we can visualize a function as a graph structure of calculations, where the input variables are represented as nodes, and each separate calculation is represented as an arrow directed to another node. These separate calculations (the arrows in the graph) are the function's elementary components.

We then are able to compute the derivatives through a process called the forward mode. In this process, after breaking down a function to its elementary components, we take the symbolic derivative of these components via the chain rule. For example, if we were to take the derivative of $sin(x)$, we would have that

![enter image description here](https://latex.codecogs.com/gif.latex?%5Cfrac&space;%7Bd%7D%7Bdx%7D&space;sin%28x%29&space;=&space;sin%27%28x%29x%27)


Where we treat “x” as a variable, and x prime is the symbolic derivative that serves as a placeholder for the actual value evaluated here. We then calculate the derivative (or gradient) by evaluating the partial derivatives of elementary functions with respect to each variable at the actual value.

For the single output case, what we actually calculate via the forward model is the product of gradient and the initializaed vector p:

![enter image description here](https://latex.codecogs.com/png.latex?D_p&space;x&space;=&space;%5Cnabla&space;x&space;%5Ccdot&space;p)

And for the multiple output case, what we actually calculate via the forward model is the product of Jacobian and the initializaed vector p:

![enter image description here](https://latex.codecogs.com/png.latex?D_p&space;x&space;=&space;J\cdot&space;p)

Thus we can obtain the gradient or Jacobian matrix of the function through different seeds of the vector p. 


## 3. How to Use *AnnoDomini*

Our package will allow the user to pass in a generic function, f. The user can then pass in any value of x for the derivative to be evaluated at. This allows the user to test out different values. Our function handles single and multi-variable inputs, as well as vectors, to make the package more generalizable. A few examples of how to implement our package are provided below.

```python
from AnnoDomini import grad
def f(x):
	return x ** 4
ad = grad(f)
print(ad(3))
>>>>>108
```
```python
# multiple input:
def f(x, y, z):
	return 2*x*y + z**4
ad = grad(f)
print(ad(1, 2, 3))
>>>>>[ 4. 2. 108.]
```

```python
# multiple input and multiple output:
def f(x, y, z):
	return [2*x*y + z**4, x*y*z]
ad = grad(f)
print(ad(1, 2, 3)) 
>>>>>[[4. 2. 108.]
	  [6. 3. 2]]
```

## 4. Software Organization

### 4.1 What will the directory structure look like?

Our directory structure will look as follows:

`AnnoDomini/` : This subdirectory will contain modules for main implementation

`tests/` : This subdirectory will contain all relevant tests

`docs/` : This subdirectory will include all relevant documentation not contained in `README.md`

`README.md` : This file will provide a brief overview of the project directory (goal, usage, structure)

`requirements.txt` : This file will list dependencies of our package.

`.gitignore`

`.travis.yml`

`license.md` : This file will contain an MIT license

### 4.2 What modules do you plan on including?

Within `AnnoDomini/`, we are planning to include 2 modules. The first module, which might be named `AutoDiff.py`, will contain implementation of the master class and its methods for calculating derivatives of elementary functions such as addition and multiplication. 

The second module, which might be named `AutoDiffExtended.py`, will potentially contain additional functions to leverage the AD module for the optimization problem (May include finding roots where the derivative/gradient equals zero) and other extensions like sampling problems (May includes methods like hamiltonian monte carlo). If we have thought of other extensions and this file to be too long, we can split the model to several submodules.

### 4.3 Where will your test suite live?

We plan to have our test suite hosted through TravisCI and CodeCov: we will run TravisCI first to test the accuracy and CodeCov to test the test coverage.

### 4.4 How will you distribute your package?

We will use Git to develop package; after we notice that the package is mature, we will follow instructions in [https://python-packaging.readthedocs.io/en/latest/index.html](https://python-packaging.readthedocs.io/en/latest/index.html) to package our code and distribute it on the PyPi.

### 4.5 How will you package your software?

Instead of using a framework such as PyScaffold, we will adhere to the proposed directory structure. We prefer to offer necessary documentation via Markdown files so they are easily readable on Github. If time permits, we also want to use the ```Sphinx``` and ```readthedocs```to publish our doumentations, since it is more user friendly.

### 4.6 Other considerations?

A potential additional feature would be to use Django or another Python web framework to provide API functionality for our feature as opposed to requiring local install and processing. This could potentially improve performance depending on the hardware we use to host this app.

## 5. Implementation

### 5.1 Core data structures

- Given that the computation table we will be constructing is inherently ordered, it makes sense to use a list or array to represent the necessary data. As we build and improve our AD implementation, we will look to optimize these structures via pre-allocation and leverage numpy arrays when possible.
- We will also be creating pandas dataframes in order to create a nice, well-structured table with good printing functionality.

### 5.2 What classes will you implement?

-   Utility class: handles input, output with proper try exceptions

```python
# Utility class
# some of the functions that want users to use
def grad(func, mode = "forward"):
	# handles args and inputs
	@functools.wraps(func)
    def wrapper(*args, **kwargs):
	    # wrapper function
        return func(*args, **kwargs).der
    return wrapper # Return the function of gradient of the func
...
```

- Automatic Differentiation Class: Similar to the implementation of dual numbers, we can calculate the forward mode of the automatic differentiation with defined AD class. Will include all necessary calculations of possible operations.

```python
# AD class
class AD:
	def __init__(self, val=0, der=1):
		self.val = val
		self.der = der
	...
```

-   Demo class:

Run some demos (include optimization demo, etc). For now we have thought of the following three demos (could be updated afterwards)

- Comparison between ad and numeric methods
- Use newton's methods to calculate the root of a given function
- Use hamiltonian monte carlo to sample from a given function

```python
class Demo():
	def compare_ad_numeric(self):
		# demo of the automatic differentiation
	def newton_method(self,func = lambda x**2 - 2*x + 1):
		# demo of the newton's method to solve the roots
	def hamiltonian_monte_carlo(self,func = lambda x: np.exp(x ** 2))
		# demo of the hamiltonian monte carlo
	...
```

### 5.3  What method and name attributes will your classes have?

#### AD class

For the core AD class, we are going to overload all possible elementary operations such as addition, or functions in numpy like np.sin, np.exp, etc. All of these functions will return another AD class with the updated value and derivative. These new operations accept numeric input as well as input with the same class.

Methods may include: 
```python
__add__
__radd__
__mul__
__truediv__
__rtruediv__
sin() # could be used for numpy sin
cos()
tan()
exp()
log()
sigmoid()
# other functions to be determined.
```

Names attribute may include:
- self.var:  the value of the calculated function (can be a vector or scalar)
- self.der: the derivative of the calculated function (can be a vector or scalar)

#### Utility class

May include several methods that users can use via our package. Currently we decide on the following methods and attributes

Methods may include: 
```python
grad() # main diff
root_finding() # newton's methods
sampler() #  hamiltonian_monte_carlo to sample
# other functions to be determined.
```

### 5.4  What external dependencies will you rely on?
- ```numpy``` and ```scipy``` for scientific calculations. We include ```scipy``` to accomodate several statistical probability functions.
- ```pandas``` is used for potential visualization since it is more user friendly.
- ```functools``` and other built in python dependencies are used to wrap functions and to manipulate built in data strutures.
- ```pytest``` is used to do the test. 

### 5.5 How will you deal with elementary functions like ```sin```, ```sqrt```, ```log```, and ```exp``` (and all the others)?

To handle these functions, we will overload these functions in the AD class, and define the updated derivative and value for the class. For instance, we may define the logarithmic function as follows:

```python
class AD()
	...
	def log(self): 
		# this would be called for numpy.log(AD)
		val = np.log(self.val)
		der = self.der * 1/self.val
		return AD(val,der)
```