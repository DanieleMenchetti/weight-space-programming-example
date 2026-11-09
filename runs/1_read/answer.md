# Reading the network: 2 -> 8 ReLU -> 1

## 1. Hidden neurons (ReLU active when pre-activation > 0); out weight in brackets

| # | Pre-activation | Active region (in words) | out w |
|---|----------------|--------------------------|-------|
| h0 | -8.27x + 8.245y - 6.256 | upper-left corner beyond the line y - x = 0.757 | -7.78 |
| h1 | -10.71x - 10.48y - 8.18 | lower-left corner beyond the line x + y = -0.77 | -8.41 |
| h2 | -0.272x + 0.19y - 0.479 | **never** (max on square = -0.017): dead neuron | +0.32 |
| h3 | -0.086x + 9.44y - 0.159 | upper half, y > ~0.017 (≈ 9.44·relu(y)) | -8.88 |
| h4 | 3.748x - 1.507y - 1.199 | right strip, x > 0.32 + 0.40y | -2.915 |
| h5 | -0.143x - 21.71y - 0.265 | lower half, y < ~-0.012 (≈ 21.7·relu(-y)) | -11.247 |
| h6 | -20.52x - 0.054y - 0.144 | left half, x < ~-0.007 (≈ 20.5·relu(-x)) | -15.628 |
| h7 | -14.69x - 7.27y + 11.77 | almost everywhere; off only in the upper-right corner (14.69x + 7.27y > 11.77) | +10.881 |

## 2. How the output combines them

logit = 0.993 + 10.881·h7 - 15.628·h6 - 11.247·h5 - 8.88·h3 - 2.915·h4 - 7.78·h0 - 8.41·h1

- h7 is a big positive "bias" (+128 at the origin) that slopes down to the right/up:
  128 - 160x - 79y.
- h6, h5, h3 are half-axis ReLUs that turn this ramp into an absolute-value shape:
  - h6 adds +321x for x<0, flipping the x slope: -160x becomes -161|x|.
  - h3 (y>0) adds -84y, giving -163y; h5 (y<0) adds +244y, giving +165y = -165|y|.
- Result, piecewise per quadrant (all ≈ 130 - 160(|x| + |y|)):
  - Q1 (x>0, y>0): 129 - 160x - 163y
  - Q4 (x>0, y<0): 132 - 158x - 165|y|
  - Q2 (x<0, y>0): 133 - 162|x| - 162y
  - Q3 (x<0, y<0): 134 - 163|x| - 166|y|
- h0, h1, h4 are small corrections. They switch on only right at the diamond's
  edge (h0 near (0, 0.8) in Q2, h1 near (0, -0.8) in Q3, h4 near (0.8, 0)) and
  pull the boundary inward a little so that every edge lands at |x| + |y| ≈ 0.80.
- h7 switches off only in the upper-right corner, which is already outside the
  diamond. There the logit is just 0.99 minus positive terms, so it stays negative.
- h2 is dead and contributes nothing.

Check: logit(0,0) = 0.993 + 10.881·11.769 ≈ +129 (class 1). At the edges:
(0.8,0) gives ≈ 0, (0,±0.8) gives ≈ 0, (-0.8,0) gives ≈ 0. At the square's
corners the logit is strongly negative.

## 3. Class-1 region

**Class 1 ⇔ |x| + |y| < 0.8**: a diamond (a square rotated 45°) centred at the
origin, with vertices at (±0.8, 0) and (0, ±0.8). Its side is ≈ 1.13 and its
area is 1.28.
