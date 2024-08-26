#!/usr/bin/env ipython


from sympy import *
import numpy as np


z1, z2, z3, z4, w0, w1, w2, w3, w4, w5, w6, w7 = symbols(
    "z1 z2 z3 z4 w0 w1 w2 w3 w4 w5 w6 w7"
)

# generating toric ideal
eqs = []
eqs = eqs + [z1**2 * z2 - w0]
eqs = eqs + [z2 - w1 * z3]
eqs = eqs + [z2**2 * z3 * z4 - w2]
eqs = eqs + [z1 * z2 * z3**3 * z4**2 - w3]
eqs = eqs + [z1 * z3 - w4]
eqs = eqs + [z1 - w5 * z2**3]
eqs = eqs + [z2 * z3 * z4**2 - w6]
eqs = eqs + [z3 * z4 - w7]

# ordering is lex because we have no objective function
result = groebner(eqs, z1, z2, z3, z4, w0, w1, w2, w3, w4, w5, w6, w7, order="lex")
result = list(result)

print(result)
print()

# extracting some solutions :
r = z1**2 * z2 * z3**2 * z4
r = r.subs({(z1, w1**3 * w5 * z3**3)})
r = r.subs({(z2, w1 * z3)})
r = r.subs({(z3 * z4, w7)})
r = r.subs({(z3**4, w4 * w1**-3 * w5**-1)})
print(r)
r = r.subs({(w1, w0 * w3 * w4**-3 * w6**-1)})
print(r)
r = r.subs({(w6, w1 * w7**2)})
print(r)
r = r.subs({(w1, w0 * w3 * w4**-3 * w6**-1)})
print(r)
