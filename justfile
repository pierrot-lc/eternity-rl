tests:
  python3 -m pytest --import-mode importlib .

trivial:
  python3 main.py \
    env=trivial \
    gamma=0.8 \
    mcts.rollouts=4 \
    model=small \
    ppo.loss.entropy_weight=1.0e-2 \
    ppo.loss.value_weight=1.0e-1 \
    ppo.rollouts=16 \
    trainer.episodes=200 \
    mode=offline

trivial_B:
  python3 main.py \
    env=trivial_B \
    model=medium \
    gamma=0.9 \
    mcts.rollouts=8 \
    model=small \
    ppo.loss.entropy_weight=1.0e-1 \
    ppo.loss.value_weight=1.0e-3 \
    ppo.rollouts=24 \
    trainer.episodes=-1 \
    mode=offline

normal:
  python3 main.py env=normal model=big mode=offline

cuda:
  python3 -c "import torch; print(torch.cuda.is_available())"

packages:
  pdm sync
