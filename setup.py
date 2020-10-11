from setuptools import setup, find_packages

setup(
    name='garage',
    version='0.1.0',
    description='Toy garage door detector',
    packages=find_packages(),
    scripts=[
        "bin/nn_infer.py",
        "bin/nn_train.py"
    ]
)