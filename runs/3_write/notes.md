# Hand-written XOR-quadrant network

Hidden neurons (all biases 0):
- h1 = relu(x + y),  h2 = relu(-x - y)  -> h1 + h2 = |x + y|
- h3 = relu(x - y),  h4 = relu(-x + y)  -> h3 + h4 = |x - y|

Output: logit = h1 + h2 - h3 - h4 = |x + y| - |x - y|  (out.bias = 0).

Why it is correct:
- |a| - |b| has the same sign as a^2 - b^2, and (x+y)^2 - (x-y)^2 = 4xy.
  So sign(logit) = sign(x*y) everywhere, not only inside [-1,1]^2.
- In closed form, logit = 2 * sign(x*y) * min(|x|, |y|).
- The zero set is exactly {x = 0} ∪ {y = 0}: the decision boundary is the two axes,
  with no approximation error. Points on the axes give logit = 0 (the spec does not
  constrain them since x*y = 0 there).
- The margin grows linearly with distance from the nearer axis (max |logit| = 2 at the corners).
  Scaling out.weight by k > 0 sharpens probabilities without moving the boundary.

Check: 20 points in 1 forward.py call (corners, near-axis points like (±0.01, 0.9),
(0.001, -1), on-axis points). Every logit had the expected sign and matched the formula.
