'''

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

'''


from setuptools import setup

setup(
    name='coala',
    version='0.1',
    description='Coala: a python package for Contextual Answer Sentence Selection (Passage Reranking)',
    url='',
    author='Lauriola Ivano',
    author_email='lauivano@amazon.com',
    license='CC-BY-NC-4.0',
    packages=['coala'],
    install_requires=['nltk',
                      'numpy',
                      'transformers==3.5',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)