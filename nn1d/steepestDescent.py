import numpy as np 

def steepestDescent(f_fun, g_fun, x0, step_size, max_iter=100, f_tol=1e-8, g_tol=1e-8):
    """
    Optimize an input function `f_fun` using steepest descent 
    x_k+1 = x_k - step_size * g(x_k)
    where g(x_k) is the gradient of f
    - f_fun: function f(x) to be optimized 
    - g_fun: gradient of f(x) 
    - x0: a numpy array for the initial guess 
    - step_size: step size/learning rate parameter
    - f_tol: absolute stopping tolerance for function |f_k+1 - f_k | < f_tol
    - g_tol: absolute stopping tolerance for gradient ||g_k+1| < g_tol 
    - save_iterates: flag for saving iterates

    Returns: 
    a list [x_0, ..., x_k], the sequence of iterates produced by the optimization algorithm
    """
    x_current = x0.copy()
    x_all = [x_current.copy()]
    f_current = f_fun(x_current)
    for k in range(max_iter + 1):
        g_current = g_fun(x_current)
        if np.linalg.norm(g_current) < g_tol:
            print("Gradient norm tolerance reached, breaking")
            break 

        x_current -= step_size * g_current
        x_all.append(x_current.copy())

        f_new = f_fun(x_current)
        if np.abs(f_new - f_current) < f_tol:
            print("Function difference tolerance reached, breaking")
            break 
        f_current = f_new 

    return np.array(x_all)