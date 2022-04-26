# Values' Projections

We first show how to produce the logits, then how to project them over the vocabulary.

### Generating GPT2 Logits

For generating value's projections, you can use our demo script which can be found [here](https://github.com/mega002/lm-debugger/blob/main/flask_server/create_offline_files.py). Specifically, you can run the `get_all_projected_values` function, found [here](https://github.com/mega002/lm-debugger/blob/main/flask_server/create_offline_files.py#L18).
Please make sure you first follow the installation requirements of the [demo repository](https://github.com/mega002/lm-debugger
).

### Generating WikiLM Logits

Similarly to GPT2, you can use the following code snippet to generate the logits for WikiLM:

```python
def get_all_projected_values(model, E):
    layer_fc2_vals = [
        model[f"decoder.layers.{layer_i}.fc2.weight"].T
        for layer_i in range(16)
    ]
    values = []
    for layer in range(16):
        for dim in range(4096):
            values.append(layer_fc2_vals[layer][dim].unsqueeze(0))
    values = torch.cat(values)
    logits = E.matmul(values.T).T.numpy()
    return logits.detach().cpu().numpy()
```

### Projecting the Logits Over the Vocabulary

After obtaining the logits, we can project them to the vocabulary and get the top-k scoring tokens, using simply:
```python
top_k = 10
projections = []
for i in range(16):
    for j in range(4096):
        d[cnt] = (i, j)
        inv_d[(i, j)] = cnt
        cnt += 1
projections = {}
for i in range(16):
    for j in range(4096):
        k = (i, j)
        cnt = inv_d[(i, j)]
        ids = np.argsort(-logits[cnt])[:top_k]
        tokens = [tokenizer._convert_id_to_token(x) for x in ids]
        projections[k] = [tokens[b] for b in range(len(tokens))]
```
This will yield the `projections` dictionary, mapping `(layer,dim)->top_k_tokens`.

For visualizing the projections, note that you can use [our demo](https://lm-debugger.apps.allenai.org/streamlit/).