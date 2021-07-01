from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='multidim_galerkin_pod',
      version='1.0.3.dev0',
      description='Helper/core functions for Galerkin POD in multiple dimensions.',
      license="GPLv3",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Heiland',
      author_email='jnhlnd@gmail.com',
      url="https://github.com/highlando/spacetime_galerkin_pod",
      packages=['multidim_galerkin_pod'],  # same as name
      install_requires=['numpy', 'scipy',
                        'sadptprj_riclyap_adi',
                        'scikit-sparse'],  # ext packages dependencies
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          ]
      )
