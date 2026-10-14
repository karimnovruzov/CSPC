# CSPC Computer Science for Physics and Chemistry
This is my pw1 repository. Each level that i made is under PW/Lab /.
## Setup
Create the environment for given lab:
```bash
conda env create -f PW/Lab\ /environment.yml
conda activate cspc
PW1
Lab A: Reproducible Foundations
What I built:
I set up the CSPC repository structure, created the conda environment, wrote unit tests for decay simulation, and compared Python loop with NumPy speed.
Speed comparison (loop vs NumPy):

    loop: 1.8910 s
    numpy: 0.0002 s
    speed-up: 9134.80x faster
Tests: There were 3 tests and all test were passed successfully
Conclusion:

I set up my first reproducible Conda environment and practiced Git branch and merge workflow.The NumPy implementation was way faster than the normal Python loop. All three pytest tests passed successfully.




## PW1 Lab B

- **What the data showed:** The dataset represents a decay process over time, showing counts decreasing steadily.
- **Observed vs Analytical:** The observed data points matched the smooth analytical decay curve (\(N_0 e^{-\lambda t}\) with \(\lambda = 0.3\)) very well.
- **Snakemake pipeline:** The Snakemake pipeline automates the workflow by running `plot.py` to generate `figure.png` from `decay_observed.csv` only when inputs change.















## PW2 Lab A: Motion from Tracking Data

### Results:





- **Mean acceleration:** -8.58 m/s²[cite: 4]
- **Standard deviation of acceleration:** 28.72 m/s²[cite: 4]
- **Integration result:** Integrating the acceleration back to position yielded values very close to the original position data.

### Noise Observation:



Differentiation magnifies noise because taking the gradient compares small, local variations between adjacent data points, making the errors larger. In contrast, integration suppresses noise because summing the values causes random errors to cancel each other out.



## PW2 - Lab B: Optimization in Chemistry

### Part 2: Optimization Methods Comparison


- Convex function f(x): All three methods (Gradient Descent, Newton, SLSQP) easily found the minimum at x ≈ 3 from start point x0 = 0.
- Complex function g(x):
  - Starting at x0 = 0, Newton's method landed on a local maximum because g''(x) < 0 at that stationary point.
  - Starting at x0 = 2, methods converged properly towards the minimum.
  - Conclusion: On a complex landscape, the choice of algorithm and initial starting point x0 critically affect the result.


### Part 3: Reaction Rate Fitting


- Fitted rate constant k = 0.25.
- The fitted decay curve matches well with the experimental data in kinetics.png.


### Part 4: Chemical Equilibrium


- Both Newton's root-finding and SLSQP minimization gave x ≈ 0.66.
- Equilibrium amounts: H2 ≈ 0.34 mol, I2 ≈ 0.34 mol, HI ≈ 1.33 mol.
- Plot saved as equilibrium.png.


- Found maximum slope using np.gradient and np.argmax.
- Equivalence point volume is approximately 50 mL.
- Plot saved as titration.png.





# Practical Work 3 (PW3) Report

## Part 1: Heart Disease Analysis (`heart.csv`)
- **Continuous Variables:** Analyzed `age`, `chol`, `trestbps`, and `thalach`.
- **Normality Test:** Shapiro-Wilk test confirmed non-normal distributions ($p \le 0.05$).
- **Group Comparison:** Mann-Whitney U test showed a significant difference in maximum heart rate (`thalach`) between diseased and healthy groups ($p \approx 0$).
- **Correlation:** Pearson correlation showed a moderate negative relationship between age and max heart rate ($r = -0.39$).

## Part 2: Chemical Exposure & Cancer Analysis (`chemicals_cancer.csv`)
- **Naive Analysis:** Initial correlation suggested Cadmium had a stronger link to tumor malignancy than Benzene.
- **Confounding Variable Control:** Filtering by `pollution_index` revealed that the apparent strong correlation of Cadmium was an artifact of confounding, identifying **Benzene** as the true primary driver.