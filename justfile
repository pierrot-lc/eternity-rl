tests:
  python3 -m pytest --import-mode importlib .

trivial:
  python3 main.py exp=trivial

trivial_B:
  python3 main.py exp=trivial_B

normal:
  python3 main.py exp=normal
