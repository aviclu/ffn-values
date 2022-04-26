# Self-Supervised Early Exit

The code in this dir reproduces the WikiLM early exit results from our paper. 

First, you will need to generate the WikiLM-based data, and install requirements. Please follow the instructions from [here](https://github.com/mega002/ff-layers#-extracting-value-and-layer-predictions).

* For running our method simulation, use the `wikilm-earlyexit-ours.ipynb` notebook.
* For running the baselines simulation, use the `wikilm-earlyexit-baselines.ipynb` notebook.

The full results table is:

| Model | accuracy | accuracy (std) | saved layers | saved layers (std) |
| --- | ----------- |----------- |----------- |----------- |
| residual vectors classifier | 94.4 | 2.1 | 3.6 | 0.5 |
| ffn vectors classifier | 92.9 |1.5 | 3.8 | 0.4 |
| ffn+residual classifier | 94.4 | 2.1 | 3.6 | 0.5 |
| ours | 94.9 | 0.6 | 3.4 | 0.1 |

