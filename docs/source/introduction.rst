Introduction
=======================================

Calculating the derivative and gradients of functions is essential to many computational and mathematical fields. In particular, this is useful in machine learning because these ML algorithms are centered around minimizing an objective loss function. Traditionally, scientists have used numerical differentiation methods to compute these derivatives and gradients, which potentially accumulates floating point errors in calculations and penalizes accuracy.

Automatic differentiation is an algorithm that can solve complex derivatives in a way that reduces these compounding floating point errors. The algorithm achieves this by breaking down functions into their elementary components and then calculates and evaluates derivatives at these elementary components. This allows the computer to solve derivatives and gradients more efficiently and precisely. This is a huge contribution to machine learning, as it allows scientists to achieve results with more precision.
