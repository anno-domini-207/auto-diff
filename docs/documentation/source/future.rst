Future Features
===============

We would like to extend our package in a few different ways:

Reverse Mode
~~~~~~~~~~~~



Second order derivative approximates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It would be beneficial to extend our package to account for calculating second derivatives, because many algorithms and methods rely on computing the second derivitve (single variable case) or the Hessian matrix (multivariable case). By handing this case,
our package will be more user-friendly and can be applied to more algorithms that require this calculation.

We could handle the single variable case as follows:

We could handle the multivariable case as follows:

Possible Applications
~~~~~~~~~~~~~~~~~~~~~

1. Hospital Scheduling (Bayesian Statistics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2. Supply Chain Management (Bayesian Statistics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3. Obesity Prevention:
^^^^^^^^^^^^^^^^^^^^^^
A relatively recent `Harvard health article <https://www.hsph.harvard.edu/obesity-prevention-source/ethnic-differences-in-bmi-and-disease-risk/>`_ found that while Asian body types are categorized as "skinny", there is a higher risk for type 2 diabetes among this race. This is because type 2 diabetes is correlated with body fat percentage. Because Asians typically have a smaller body size, they are mistaken for being "healthy" even with poor eating habits.
Consequently, those with poor eating habits could have a larger percentage of body fat, but it is disguised as "skinny fat". It would be interesting to model the body statistics (i.e. BMI, body fat percentage, height, weight, etc.) of this particular subgroup and determine the optimal diet plan to help reduce their risk of type 2 diabetes. This could be done by using an optimization algorithm to find the amount of nutrients that could help their body quickly recover. This problem could also involve modeling how their body reacts
to certain nutrients, and finding the optimal amount of specified nutrients that could help reduce their body fat percentage. All of which would need to derive gradients within the algorithms used.
