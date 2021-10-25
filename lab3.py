'''
This code is some demonstration
of my skills in using mathematical libraries to perform calculations in Python
'''

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# task1 function is performing calculations using secant method.
def task1(a, b, expr):
    x = sym.Symbol('x')
    x0 = a
    x1 = b
    f = sym.lambdify(x, expr)
    x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    eps = 0.001
    i = 0
    while abs(x2 - x1) >= eps:
        i = i + 1
        x0 = x1
        x1 = x2
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    return x2, i
 
# task2 function is performing calculations using Newton's method.
def task2(a, b, f):
    print(f)
    x = sym.Symbol('x')
    x0 = (a + b) / 2
    eps = 0.001
    x0 = a if sym.diff(f, x, 1).subs(x, a) * f.subs(x, x0).evalf() > 0 else b
    x1 = x0 - f.subs(x, x0) / sym.diff(f, x, 1).subs(x, x0)
    i = 0
    while abs(x1 - x0) >= eps:
        i = i + 1
        x0 = x1
        x1 = x0 - f.subs(x, x0).evalf() / sym.diff(f, x, 1).subs(x, x0)
    return x1, i

# task2 function is performing calculations using simple iteration method.
def task3(a, b, phi):
    eps = 0.0001
    x0 = (a + b) / 10
    q = (sym.diff(phi, x, 1).subs(x, x0) + 1)
    print(q)
    i = 0
    x1 = phi.subs(x, x0).evalf()
    while abs(x1 - x0) >= (eps * (1 - q) / q):
        i = i + 1
        x0 = x1
        x1 = phi.subs(x, x0).evalf()
        print("x0: ", x0, "\n")
        print("x1: ", x1, "\n")
    return x1, i

'''
The code below is an application of the methods described
in the functions above for specific equations.
'''

# task 1
x = sym.Symbol('x')

xnp1 = np.arange(-5, 5, 0.5)
eq1 = (x - 1) * (x - 1) - 0.5 * sym.exp(x)
ynp1 = np.zeros(len(xnp1))
for i in range(len(ynp1)):
    ynp1[i] = eq1.subs(x, xnp1[i])
plt.plot(xnp1, ynp1)
plt.show()
xst1, iter1 = task1(-3, 3, eq1)
print('x* = ', xst1)
print('Number of iterations is ', iter1)
print('eq1 tochnoe - eq1: ', abs(0.213309 - xst1))

# task 2
xnp2 = np.arange(-10, 20, 0.5)
eq2 = 2*x - 1.3 ** x
ynp2 = np.zeros(len(xnp2))
for i in range(len(ynp2)):
    ynp2[i] = eq2.subs(x, xnp2[i])
plt.plot(xnp2, ynp2)
xst2, iter2 = task2(-1, 3, eq2)
print('x* = ', xst2)
print('Number of iterations is ', iter2)
print('eq2 tochnoe - eq2: ', abs(0.582573 - xst2))
plt.show()

# task 3
xnp3 = np.arange(0, 1, 0.01)
eq3 = sym.log(x) + (x+1)**3
ynp3 = np.zeros(len(xnp3))
for i in range(len(ynp3)):
    ynp3[i] = eq3.subs(x, xnp3[i])
plt.plot(xnp3, ynp3)
phi = sym.exp(-(x+1)**3)
xst3, iter3 = task3(0, 3, phi)
print('x* = ', xst3)
print('Number of iterations is ', iter3)
print('eq3 tochnoe - eq3: ', abs(0.187439 - xst3))
plt.show()
