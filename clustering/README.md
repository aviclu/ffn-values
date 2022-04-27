# Values' Clustering

The clustering procedure is identical for both GPT2 and WikiLM. Follow the following instructions in order to produce the clusters:

1. You should first produce the logits (not the projections). You can use the instructions and code under the `values_projections` dir.
2. Given the logits, you should compute their pairwise cosine distance matrix. You can use [this code](https://github.com/mega002/lm-debugger/blob/main/flask_server/create_offline_files.py#L65).
3. Then apply agglomerative clustering over the distance matrix. You can use [this code](https://github.com/mega002/lm-debugger/blob/main/flask_server/create_offline_files.py#L72).

For visualizing the clusters, note that you can use [our demo](https://lm-debugger.apps.allenai.org/streamlit/).
