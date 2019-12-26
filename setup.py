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
        'pandas==0.25.3',
        'matplotlib==3.0.3',
        'scipy==1.2.1',
        'pillow==6.2.1',
        'imageio==2.6.1',
        'WeasyPrint==50'
    ]
)
