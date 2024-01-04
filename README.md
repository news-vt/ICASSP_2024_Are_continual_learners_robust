ICASSP 2024: Analysis of  the Memorization and Generalization Capabilities of AI Agents: Are Continual Learners Robust?

To run experiments:

Use `./utils/main.py --seed 0 --dataset rot-mnist --model eqrm --lr 0.1 --n_epochs 1 --batch_size 512 --eqrm 0.9999 --env_batch 3 --balance 0.5 --heldout 6 --minibatch_size 64` to run experiments.

arguments explanation: 

  batch_size: batch size from data stream
  
  eqrm: alpha in our formulated problem
  
  env_batch: the number of batches to estimate the environemtal distribution
  
  heldout: the index of leftout roation to test the generalization 0 -> 0, 1 -> 25, 2 -> 50, ...,               6 -> 150
  
  minibatch_size: the size of batches to estimate the environmental distribution

We used the framework of https://github.com/aimagelab/mammoth for our experiments

