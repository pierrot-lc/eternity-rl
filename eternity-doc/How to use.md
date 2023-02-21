# Testing
Tests are done using `pytest`.
To allow relative imports, you need to set the flag `--import-mode importlib`, and to call the program explicitly from the `python -m` argument.

 So to check if all tests are passing, you need to use the following command in the root directory :

```sh
python3 -m pytest --import-mode importlib .
```