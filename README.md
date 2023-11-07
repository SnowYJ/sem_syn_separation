Implementation of work "Graph-induced Syntax and Semantic Separation in Variational AutoEncoders"

![encoding overview](sem_syntax.png)

***
running encoding baselines:

1. train_optimus_disentangle_graph.py
2. train(evaluate)_optimus_disentangle_lstm.py
3. train(evaluate)_optimus_disentangle_siam.py
4. train(evaluate)_optimus_separate_graph.py
***

![decoding overview](overview.png)

***
running decoding baselines:

1. train(evaluate)_optimus_separate_graph_syntax_constraint_gpt2.py

```python
fuse_way = "add_syn_Q_sem_KV"
# add_syn_Q_sem_KV
# add_syntax_query
# fuse_syntax_query
# fuse_syn_Q_sem_KV
```
***
