tests:
  python3 -m pytest --import-mode importlib .

trivial:
  python3 main.py --config-name trivial mode=offline

trivial_B:
  python3 main.py --config-name trivial_B mode=offline

normal:
  python3 main.py --config-name normal mode=offline

cuda:
  python3 -c "import torch; print(torch.cuda.is_available())"

packages:
  pdm sync
