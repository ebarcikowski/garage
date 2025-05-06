from setuptools import setup, find_packages

setup(
    name='garage',
    version='0.1.0',
    description='Toy garage door detector',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train=garage.train:main',
            'infer=garage.infer:main',
            'app=garage.app:main',
        ]
    }
)