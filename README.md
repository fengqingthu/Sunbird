# <img src="https://github.com/fengqingthu/Sunbird/blob/main/logo.png?raw=true" width="128" height="128"> Sunbird: A Solar Chimney Optimizer 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Instructions

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Clone this repository

   ```bash
   $ git clone https://github.com/fengqingthu/Sunbird.git
   ```

3. Navigate into the project directory

   ```bash
   $ cd Sunbird
   ```

4. Create a new virtual environment

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Run the app

   ```bash
   $ flask run
   ```

7. You should be able to access the APIs using [Hops](https://developer.rhino3d.com/guides/compute/hops-component/) at http://localhost:5000/predict and http://localhost:5000/optimize. See also this [example.gh](https://github.com/fengqingthu/Sunbird/blob/main/example.gh).

## Data & Training
Our [dataset](https://github.com/fengqingthu/Sunbird/blob/main/data/purged.csv) and [training script](https://colab.research.google.com/drive/1IFtweQr6FRN_HqoQ-B03fmwA4TwQNhnl#scrollTo=idxsdXQyYKTo) are also included for your reference in case you want to import your own data.