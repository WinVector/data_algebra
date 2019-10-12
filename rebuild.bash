
bash ./clean.bash

pytest --cov data_algebra tests > coverage.txt
cat coverage.txt
python3 setup.py sdist bdist_wheel
pip install dist/data_algebra-*.tar.gz
twine check dist/*
