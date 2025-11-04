# object_scores

## Installation

In a terminal, make a virtual environment and activate it. For this example, I'll call it `object`. To make the environment on Linux/Mac, run

``` bash
python -m venv object
source object/bin/activate
```

...or on Windows, run

``` bash
python -m venv object
source object\script\activate.bat
```

Now clone (download) this repo, move inside the folder you download and install the package:

``` bash
git clone https://github.com/chrishalcrow/object_scores.git
cd object_scores
pip install -e .
```

Now you should be able to open the intro script by running

``` bash
jupyter lab scripts/intro.ipynb
```