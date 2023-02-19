The goal of this phase is to have a model already acting well so that it can be finetuned easily.
The data is a solved random instance, from a starting point to the final solved state.
The model is trained using imitation learning.

# Data Generation

The data is generated using the following scheme:

1. Generate a random solved instance.
2. Apply random actions to degrade its state.
3. Compute the inverse actions in order to get back to the solved state.
4. Save the degraded instance along with the actions that solved it.
