tests:
  python3 -m pytest --import-mode importlib .

trivial:
  python3 main.py exp=trivial mode=offline

trivial_B:
  python3 main.py exp=trivial_B mode=offline

normal:
  python3 main.py exp=normal mode=offline

cuda:
  python3 -c "import torch; print(torch.cuda.is_available())"
