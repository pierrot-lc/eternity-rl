tests:
  python3 -m pytest --import-mode importlib .

trivial:
  python3 main.py env=trivial model=small mode=offline

trivial_B:
  python3 main.py env=trivial_B model=medium mode=offline

normal:
  python3 main.py env=normal model=big mode=offline

cuda:
  python3 -c "import torch; print(torch.cuda.is_available())"
