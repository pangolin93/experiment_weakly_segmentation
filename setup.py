import os

from setuptools import setup, find_packages
from setuptools.command.develop import develop as Develop
from setuptools.command.install import install as Install


def _get_resources(package_name):
    # Get all the resources (also on nested levels)
    res_paths = os.path.join(package_name, "resources")
    all_resources = [os.path.join(folder, file) for folder, _, files in os.walk(res_paths) for file in files]
    # Remove the prefix: start just from "resources"
    return [resource[resource.index("resources"):] for resource in all_resources]


# Package configuration
setup(name='weakseg',
      version='0.0.0',
      description='I wanna play with Weakly Supervised Segmantation',
      include_package_data=True,
      setup_requires=["wheel"],
      packages=find_packages(),
      package_data={
          "weakseg": _get_resources("weakseg")
          }
    )