"""Install nglib dependencies and register it globally.

For installing test dependencies run: python3 setup.py test

"""

from setuptools import setup, find_packages

setup(name='tpy',
      version='0.1.0',
      description='Transport in Python',
      author='Gonzalo Rios',
      author_email='grios@dim.uchile.com',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib', 'seaborn', 'torch', 'pyro-ppl','dill'],
      tests_require=['pytest', 'pytest-mpl'
      ],
      zip_safe=False)
