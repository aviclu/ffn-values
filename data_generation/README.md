# Data Generation

For generating the WikiLM-based data, please follow the instructions from [here](https://github.com/mega002/ff-layers#-extracting-value-and-layer-predictions).

### Generating GPT2-based data
Set up the environment, run the following commands, which install the required packages:

```
pip install -r requirements.txt
python -m spacy download en
```

To generate data, run the `generate_examples.py` script (see the help menu for configuration options).

The code was tested in a Python 3.7 environment.