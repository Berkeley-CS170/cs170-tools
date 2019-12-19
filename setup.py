from setuptools import setup, find_packages
setup(
    name="cs170-tools",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cs170-grading = cs170.grading.script:main'
        ]
    },
    install_requires=[
        'numpy==1.15.0',
        'pandas==0.23.4',
        'matplotlib==3.0.3',
        'scipy==1.2.1',
        'pillow==6.0.0',
        'imageio==2.6.1'
    ]
)
