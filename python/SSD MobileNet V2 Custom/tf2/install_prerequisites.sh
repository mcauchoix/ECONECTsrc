pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

cp object_detection/packages/tf2/setup.py .
python -m pip install --user --use-feature=2020-resolver .
pip install --upgrade tensorflow-gpu==2.5.0
pip install IPython
pip install seaborn
pip install numpy==1.17.3 --user
pip install -U scikit-learn
pip install pandas