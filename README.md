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