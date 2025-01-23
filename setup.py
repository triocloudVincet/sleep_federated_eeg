from setuptools import setup, find_packages

setup(
    name="sleep_federated",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scipy>=1.10.0',
        'mne>=1.5.0',
        'wandb>=0.15.0',
        'wget>=3.2',
    ]
)
