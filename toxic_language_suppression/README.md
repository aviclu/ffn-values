# Toxic Language Suppression

The code in this directory reproduces the GPT2 Zero-Shot Toxic Language Suppression results from our paper. 


<p align = "center">
<img width="618" alt="Screen Shot 2022-04-15 at 1 25 51 PM" src="https://user-images.githubusercontent.com/50050060/163601704-12852437-23cc-4e96-8b40-3b0b3d6946a3.png">
</p>

You will first need to download the Real Toxicity Prompts, which you find [here](https://allenai.org/data/real-toxicity-prompts).

You can reproduce our results and the word-filter baseline through the `Toxicity-Suppression.ipynb` notebook. You can also reproduce our results as a command-line utility. 

To reproduce the self-debiasing results, please go to the [self-debiasing repo](https://github.com/timoschick/self-debiasing) and follow the instructions there. 

## Command Line Utility
To reproduce the perplexity results, run:
```
python3 perplexity.py --model_name gpt2-medium \
                      --values_filepath non_toxic_values.pickle \
                      --coef_value 3 \
                      --use_cuda
```

To reproduce the 10 manual pick values results, run:
```
python3 toxicity_scoring.py --prompts_filename prompts.jsonl \
                            --output_dir toxicity-suppresion-results \
                            --api_key <API_KEY> \
                            --model_name gpt2-medium \
                            --values_filepath non_toxic_values.pickle \
                            --challenging_only \
                            --coef_value 3 \
                            --mode toxic-suppr \
                            --use_cuda
```

To reproduce the word-filter results, run:
```
python3 toxicity_scoring.py --prompts_filename prompts.jsonl \
                            --output_dir toxicity-suppresion-results \
                            --api_key <API_KEY> \
                            --model_name gpt2-medium \
                            --challenging_only \
                            --mode word-filter \
                            --use_cuda
```

