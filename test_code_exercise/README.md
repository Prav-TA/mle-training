## How to Run This Project


### Step 1: Create the conda environment in the folder

`$ cd test_code_exercise`

`$ conda env create -f environment.yml`

The above command should create a conda python environment named `testcase-dev`

### Step 2: Activate the environment
`$ conda activate testcase-dev`


- Run:
`$ python -m pip install -e . &&     find . -type d -name "__pycache__" -exec rm -rf {} + &&     find . -type d -name "*.egg-info" -exec rm -rf {} + &&     rm -rf build`

  to have `test_archive` installed in the env from the code folder and also help cleaning unwanted cache files

### Step 3: Run pystest on test_analysis.py to run the test cases

`$ pytest test_archive/tests/unittests/tsttemplate/test_analysis.py -v`