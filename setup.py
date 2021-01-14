from setuptools import setup, find_packages
import os

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as fb:
        requirements = fb.readlines()
else:
    requirements = []

print(find_packages())
setup(
    name="PongBehavior",
    version="0.1",
    packages=['PongBehavior'],
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=requirements,
    # metadata to display on PyPI
    author="Rishi Rajalingham",
    author_email="rishir@mit.edu",
    description="Analysis for pong task, behavior",
    keywords="PongBehavior",
    # could also include long_description, download_url, etc.
)

setup(
    name="PongDatasets",
    version="0.1",
    packages=['PongDatasets'],
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=requirements,
    # metadata to display on PyPI
    author="Rishi Rajalingham",
    author_email="rishir@mit.edu",
    description="Analysis for pong task, behavior",
    keywords="PongDatasets",
    # could also include long_description, download_url, etc.
)

setup(
    name="PongRNN",
    version="0.1",
    packages=['PongRNN'],
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=requirements,
    # metadata to display on PyPI
    author="Rishi Rajalingham",
    author_email="rishir@mit.edu",
    description="Analysis for pong task, behavior",
    keywords="PongRNN",
    # could also include long_description, download_url, etc.
)