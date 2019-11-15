# this is our main class. Will contain implementation of the master class and its methods for calculating derivatives of elementary functions such as addition and multiplication.

import numpy as np

class AD:
    def __init__(self, val=0.0, der=1.0):
        self.val = val
        self.der = der

    def __repr__(self):
        return f'Function Value: {self.val} | Derivative Value: {self.der}'

    def __add__(self, other):
        try:
            val = self.val + other.val
            der = self.der + other.der
        except AttributeError:
            val = self.val + other
            der = self.der
        return AD(val, der)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            val = self.val - other.val
            der = self.der - other.der
        except AttributeError:
            val = self.val - other
            der = self.der
        return AD(val, der)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        try:
            val = self.val * other.val
            der = self.der * other.val + self.val * other.der  # By product rule
        except AttributeError:
            val = self.val * other
            der = self.der * other
        return AD(val, der)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if (other == 0) or (other.val == 0):
            raise ZeroDivisionError
        try:
            val = self.val / other.val
            der = ((self.der * other.val) - (other.der * self.val)) / (other.val ** 2)  # By quotient rule
        except AttributeError:
            val = self.val / other
            der = self.der / other
        return AD(val, der)

    def __rtruediv__(self, other):
        # Here, we only need to consider the case when `other` (numerator) is a number
        # If `other` is an AD object, its __truediv__ method takes care of things
        if self.val == 0:
            raise ZeroDivisionError
        val = other / self.val
        der = -(other * self.der) / (self.val ** 2)  # By chain/quotient rule
        return AD(val, der)

    def __pow__(self, n):
        val = self.val ** n
        der = n * (self.val ** (n - 1)) * self.der
        return AD(val, der)

    def __rpow__(self, n):
        if n > 0:
            val = n ** self.val
            der = (n ** self.val) * np.log(n) * self.der
        elif n < 0:
            raise ValueError("Domain error: Logarithm of a negative number cannot be evaluated!")
        else:  # n == 0
            if self.val > 0:
                val = 0
                der = 0
            elif self.val < 0:
                raise ZeroDivisionError
            else:  # self.val == 0
                val = 1
                der = 0
        return AD(val, der)

    def __neg__(self):
        val = -self.val
        der = -self.der
        return AD(val, der)
