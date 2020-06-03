from setuptools import setup

setup(name='mm2d',
      version='0.1',
      description='Simulation of two-dimensional mobile manipulator.',
      author='Adam Heins',
      author_email='mail@adamheins.com',
      install_requires=['numpy', 'matplotlib', 'cython'],
      packages=['mm2d'],
      python_requires='>=3',
      zip_safe=False)
