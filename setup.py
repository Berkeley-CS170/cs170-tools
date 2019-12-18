from setuptools import setup, find_packages
setup(
    name="cs170_grading",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cs170-grading = cs170_grading.__main__:main'
        ]
    },
    install_requires=[
        'numpy==1.15.0',
        'pandas==0.23.4',
        'matplotlib==3.0.3'
    ]
)
