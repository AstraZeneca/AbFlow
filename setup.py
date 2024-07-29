#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="abflow",
        version="0.0.1",
        packages=find_packages(),
        author="Haowen Zhao",
        author_email="hz362@cam.ac.uk",
        description="Abflow: Structure-sequence codesign model for CDR redesign",
        include_package_data=True,
    )
