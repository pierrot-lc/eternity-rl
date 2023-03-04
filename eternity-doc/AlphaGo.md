# Etapes
1. Supervised initialisation (rollout & policy networks)
2. RL of pools of networks that play against each other (policy & value)
3. Use value network for MCMT

# Networks
1. Policy network (CNN)
2. Rollout policy (Linear)
3. Value network (CNN)

# Supervised
From 30M states of pro players.
55% d'accuracy pour le gros modèle (CNN).
Rollout et policy sont entraînés de la même manière.

# RL
Each time a game finishes, we sample one state of the game and the model has to predict the end value of the game.

Value network : Trained using MSE

# MCTS
Utiliser le value network pour l'évaluation d'un noeud, utiliser la rollout policy pour dive un noeud et estimer sa valeur en fonction de la valeur finale du dive.

Pour étendre un noeud, utiliser le value network.