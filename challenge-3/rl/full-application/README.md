# Generate, Evaluate, Retrain Concurrently

This application gradually improves our ability to generate water clusters by continually training an RL policy,
validating the clusters predicted by RL,
and updating the reward function used by RL using the validation results.

To do list: 
- [ ] Add a model training function
- [ ] Feed the model training back into the RL policy 
- [ ] De-duplicate list of graphs to be evaluated
- [ ] Push updated results to the MongoDB
