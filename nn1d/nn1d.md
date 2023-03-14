# NN1D: investigating neural networks in 1D 


## Mar 10
### 1. Steepest Descent Implementation 
Recall the algorithm/pseudo code we wrote for steepest descent. Complete the implementation in `steepestDescent.py` to solve the optimization problem
$$\min_{x} f(x)$$ 
given the function `f_fun` and its gradient `g_fun`. Note, here `f_fun` is a function that takes in a vector $x$ (numpy array) of size $n \geq 1 $, and the gradient `g_fun` takes in $x$ and returns something the same size as $x$. 

### 2. Optimization on quadratics
Test the algorithm on a quadratic function 
$$ f(x) = a_1 x_1^2 + a_2 x_2^2 $$ 
where $x = (x_1, x_2)$ is a point in 2D, and the coefficients $a_1, a_2 > 0$. Note that this has the optimum at $x^* = (0,0)$. You can check if your implementation converges to this optimum.

I have written the quadratic function in `optimize_quadratic.py`. You can import `steepestDescent` from your own implementation and use it in `optimize_quadratic.py`. 

## Feb 18 
### 1. Plotting some 2D functions in numpy 
This week we will try and plot some 2D functions $f(x_1, x_2)$ (or maybe you want to write this as $f(x,y)$). Look at the example in `plotting_2d.py`. Here we have $f(x, y) = x^2 + y^2$. There are a couple of functions being used 
1. `xgv = np.linspace(-1, 1, n_points)` creates a 1D [-1, ..., 1] with number of points equal to `n_points` 
2. `X, Y = np.meshgrid(xgv, ygv)` takes two 1D grids and creates a 2D grid corresponding to the $x$ and $y$ values
3. `Q = quadratic(X,Y)` evaluates the quadratic function at each pair of $x,y$ points on the grid
4. `plt.contourf` or `ax.plot_surface()` can plot the function of 2 variables given the `x,y` grid and the values $f(x,y)$ on the grid. 

Try look into the example and check that you understand what's going on
1. See what `X`, `Y`, `Q` look like. You can reduce the `n_points` and print out their values. 
2. Try plotting some other 2D functions, e.g. $f(x,y) = sin(x) sin(y)$ over the box $-10 \leq x \leq 10$ and $-10 \leq y \leq 10$.

You can just play around with the code and add this to it. 

### 2. Some multivariable calculus revision
Remind yourself of the gradient, $\nabla f$, of a multivariable function $f(x,y)$. How do you compute it and what does it mean? What about functions with $n$ inputs, $f(x_1, x_2, \dots, x_n)$?

Next time we will start trying to optimize multivariable functions.

## Feb 10

We are interested in seeing how well neural networks can approximate functions $y = f(x)$ in 1D, i.e. $x \in \mathbb{R}$  And $y \in \mathbb{R}$ . Here are some things to do for this week 

### 1. Plotting functions using numpy
Try to plot some functions in numpy, which will give you some practice with numpy arrays and functions. Here are some examples to get you started 
- $y = x^{p}$ for a few different values of $p$ 
- $y = sin(kx), \quad y = cos(kx), \quad y = tan(kx)$ For some different $k$ values
- $y = e^{x}, \quad y = ln(x)$ 
- And some kind of combination, e.g. $y = 3 sin^2(10x) + x^2 - e^{-x}$

Try some different ranges, e.g. $x \in [0, 1], [-1, 1], [-10, 10]$. 

Some notes 
- Most mathematical operations in numpy can be done with arrays, usually element wise. That means if `x` is an array, then `x**2` will square every element of `x`. Similarly, if `y` is an array of the same length, `x + y` will add each element.
- You can look back to our code and the tutorial 
- Remember, once you plot the graph, the code needs to have `plt.show()` in order for the graph to render. 

### 2. Trying some simple connections with single neurons

Connect a few neurons together and see what the functions looks like. Does anything interesting show up? You can run these multiple times using randomized weights.

1. A sequence of neurons, where the last one is a linear neuron (no activation function). 
2. What about if you have two different neurons using the same input value, i.e. $a_1 = \sigma(w_1 x + b_1)$ and $a_2 = \sigma(w_2x + b_2)$ and then add their outputs together as your final output, i.e. $y = a_1 + a_2$
3. Try number 2 but using different weightings on $a_1$ and $a_2$, i.e. we have new weights $v_1, v_2$, to output $y = v_1 a_1 + v_2 a_2$. You can choose random $v_1$ and $v_2$ as well.
4. What about adding together $n$ neurons? That is, $a_1 = \sigma(w_1 x + b_n)$ all the way up to $a_n = \sigma(w_n x + b_n)$, and then summing them together with different weightings $y = v_1 a_1 + \dots + v_n a_n$.
### 3. Rewriting the multi-neuron sum as its own class 

Instead of just a single `Neuron` class, let’s now write a class to do what number 4 does. The class should take in 
- An array of weights for the neurons $w = [w_1, \dots w_n]$
- An array of biases for the neurons $b = [b_1, \dots, b_n]$
- An array of weights for the output $v = [v_1, \dots v_n]$ 
- The choice of activation function to use. Let’s just keep it as `sigmoid` or `relu` for now. 

And then have an evaluation method for computing the output 

$$ y = \sum_{i=1}^{n} v_i a_i = v_1 a_1 + \dots + v_n a_n $$
Where $a_i$ is the output of each neuron from the input
$$ a_i(x) = \sigma(w_i x + b_i) $$

I have made a new bit of code in the same folder as the NN code today. It’s called `singleLayerNetwork1D.py`. There’s a class `SingleLayerNetwork` with the inputs and methods sketched out. See if you can fill that in to do the evaluation. 

Now try and plot these for some different weights, and in particular, different `n` values. How do the functions look? Save some screenshots of plots that look interesting

Tips:
- Remember you can generate entire random arrays by either `np.random.randn(n)` or `np.random.rand(n)`.
- Numpy Arrays can be used just like lists. You can access elements by `myArray[i]`. You can also loop through them like lists: `for myItem in myArray`. You can get the length of an array like lists: `len(myArray)`. See the tutorial or ask me if you have any questions
