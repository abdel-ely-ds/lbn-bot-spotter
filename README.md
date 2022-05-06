LBN BOT SPOTTER
==============================

A Linear Model For Detecting Fake Users In LBN Website

## Getting Started
Installation
------------
    $ git clone https://github.com/abdel-ely-ds/lbn-bot-spotter.git
    $ cd lbn-bot-spotter
    $ pip install .

Usage
------------

```python
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

from bot_spotter import Spotter
from bot_spotter.constants import DATA_FILE, DATA_FOLDER, LABEL

raw_data = pd.read_csv(os.path.join("../", DATA_FOLDER, DATA_FILE))
train, val = train_test_split(random_state=0, test_size=0.2)

spotter = Spotter()
spotter.run(train)
preds = spotter.predict(val)
f1_score(val[LABEL], preds)
```

From CLI
------------
    $ bot-spotter --output-dir ./artifact --input-file-path ./data/fake_users.csv

Project Organization
------------

    ├── LICENSE            <- MIT License.
    ├── README.md          <- README of the project.
    ├── data               <- Raw data
    ├── src                <- source code for training and predicting bots
    ├── notebooks          <- Jupyter notebooks
    ├── requirements.txt   <- The requirements file contains all the necessary libs to run the project
    ├── src                <- Source code for the project.
    ├── tests              <- tests forlder
    └── noxfile.py          <- black, build, tests               

--------
