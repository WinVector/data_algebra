
bash ./clean.bash

# pytest --cov data_algebra tests > coverage.txt
pip install --no-deps -e "$(pwd)"
pytest --cov-report term-missing --cov data_algebra tests > coverage.txt
pdoc3 --force -o ./docs ./data_algebra
cat coverage.txt
python3 setup.py sdist bdist_wheel
# pip install dist/data_algebra-*.tar.gz
twine check dist/*

