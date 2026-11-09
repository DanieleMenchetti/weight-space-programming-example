# Repair notes

## Corrupted weight
`out.weight[6]` (output weight of hidden neuron h6) had its sign flipped: it was **+15.628**, and the fix is **-15.628**.
This is the only number changed.

## How I found it
- I read each hidden unit as a feature:
  - h6 = relu(-20.5x - 0.14) ≈ 20.5·relu(-x), which measures "how far left" a point is.
  - h3 ≈ relu(y) and h5 ≈ relu(-y) measure "how far up" and "how far down".
  - h0 and h1 are the top-left and bottom-left diamond edges, and h4 is the right side.
  - h7 is a large-bias "inside" unit.
- To fit |x|+|y| < 0.8, every distance-from-centre feature must push the logit down. All of them have negative
  output weights (-7.78, -8.41, -8.88, -2.915, -11.247) except h6 (+15.628), which pushed the logit *up* the further left a point was.
- All 8 bug-report points have x < 0, which matches an h6 fault.
- Hand check at (-0.37, -0.60): h6 = 7.45, so its contribution is +116 and the logit is about +191 (wrong).
  With -15.628 the logit is about -42 (correct).

## Why the fix is correct
- With the sign restored, the logit along the -x axis is about 131 - 161·|x| (plus the h0/h1 edge terms), so it
  crosses 0 near |x| ≈ 0.8. This mirrors the +x side, where h7 and h4 do the same job.
- I made 2 forward.py calls on repaired.json (64 points), and every point was classified correctly:
  - All 8 bug-report points are now class 0 (logits -4 to -180).
  - Points just inside (|x|+|y| = 0.76) and just outside (= 0.84) the boundary were correct on all four edges and both
    axes. The logit changes sign between them every time (e.g. (-0.38,0.38) gives +9.5 and (-0.42,0.42) gives -8.5).
  - The centre, the corners and other interior and exterior points were also correct.
