from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sumtyme',
    version='1.0.9',
    author='sumteam',
    author_email='team@sumtyme.ai',
    description='Python Client for Embedded Intelligence Platform by sumtyme.ai.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sumteam/sumtyme-eip-client', 
    packages=find_packages(), 
    install_requires=requirements, 
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable', 
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.9', 
)