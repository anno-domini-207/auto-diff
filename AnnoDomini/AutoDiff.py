# The AutoDiff class implements methods for (automatically) calculating derivatives of elementary functions.
# As of now (11/15/2019), the implementation is intended and works only for scalar functions of a single input,
# but the functionality will be expanded in coming weeks to support multiple functions of multiple inputs.

import numpy as np


class AutoDiff:
    def __init__(self, val=0.0, der=1.0):
        self.val = val
        self.der = der

    def __repr__(self):
        return f'Function Value: {self.val} | Derivative Value: {self.der}'

    def __eq__(self, other):
        return (self.val == other.val) and (self.der == other.der)

    def __ne__(self, other):
        return (not self == other)

    def __add__(self, other):
        try:
            val = self.val + other.val
            der = self.der + other.der
        except AttributeError:
            val = self.val + other
            der = self.der
        return AutoDiff(val, der)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            val = self.val - other.val
            der = self.der - other.der
        except AttributeError:
            val = self.val - other
            der = self.der
        return AutoDiff(val, der)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        try:
            val = self.val * other.val
            der = self.der * other.val + self.val * other.der  # By product rule
        except AttributeError:
            val = self.val * other
            der = self.der * other
        return AutoDiff(val, der)

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
        return AutoDiff(val, der)

    def __rtruediv__(self, other):
        # Here, we only need to consider the case when `other` (numerator) is a number
        # If `other` is an AD object, its __truediv__ method takes care of things
        if self.val == 0:
            raise ZeroDivisionError
        val = other / self.val
        der = -(other * self.der) / (self.val ** 2)  # By chain/quotient rule
        return AutoDiff(val, der)

    def __pow__(self, n):
        val = self.val ** n
        der = n * (self.val ** (n - 1)) * self.der
        return AutoDiff(val, der)

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
        return AutoDiff(val, der)

    def __neg__(self):
        val = -self.val
        der = -self.der
        return AutoDiff(val, der)

    def sqrt(self):
        return AutoDiff(self.val, self.der) ** 0.5

    def sin(self):
        val = np.sin(self.val)
        der = np.cos(self.val) * self.der
        return AutoDiff(val, der)

    def cos(self):
        val = np.cos(self.val)
        der = -np.sin(self.val) * self.der
        return AutoDiff(val, der)

    def tan(self):
        if ((self.val / np.pi) - 1 / 2) % 1 == 0:
            raise ValueError("Domain error: tangent is undefined at (1/2+k)*pi where k is any integer!")
        if np.cos(self.val) == 0:
            raise ZeroDivisionError
        val = np.tan(self.val)
        der = ((1 / np.cos(self.val)) ** 2) * self.der
        return AutoDiff(val, der)

    def arcsin(self):
        if not (-1 <= self.val <= 1):
            raise ValueError("Domain error: arcsin is defined at [-1, 1]!")
        if np.abs(self.val) == 1:
            raise ZeroDivisionError
        val = np.arcsin(self.val)
        der = (1 / np.sqrt(1 - (self.val ** 2))) * self.der
        return AutoDiff(val, der)

    def arccos(self):
        if not (-1 <= self.val <= 1):
            raise ValueError("Domain error: arccos is defined at [-1, 1]!")
        if np.abs(self.val) == 1:
            raise ZeroDivisionError
        val = np.arccos(self.val)
        der = (-1 / np.sqrt(1 - (self.val ** 2))) * self.der
        return AutoDiff(val, der)

    def arctan(self):
        val = np.arctan(self.val)
        der = (1 / (1 + (self.val) ** 2)) * self.der
        return AutoDiff(val, der)

    def sinh(self):
        val = np.sinh(self.val)
        der = np.cosh(self.val) * self.der
        return AutoDiff(val, der)

    def cosh(self):
        val = np.cosh(self.val)
        der = np.sinh(self.val) * self.der
        return AutoDiff(val, der)

    def tanh(self):
        if np.cosh(self.val) == 0:
            raise ZeroDivisionError
        val = np.tanh(self.val)
        der = (1 / (np.cosh(self.val) ** 2)) * self.der
        return AutoDiff(val, der)

    def log(self):
        if self.val <= 0:
            raise ValueError("Domain error: Logarithm is defined for positive numbers only!")
        if self.val == 0:
            raise ZeroDivisionError
        val = np.log(self.val)
        der = (1 / self.val) * self.der
        return AutoDiff(val, der)

    def exp(self):
        val = np.exp(self.val)
        der = np.exp(self.val) * self.der
        return AutoDiff(val, der)

    def logistic(self):
        val = 1 / (1 + np.exp(-self.val))
        der = np.exp(self.val) / ((1 + np.exp(self.val)) ** 2)
        return AutoDiff(val, der)
