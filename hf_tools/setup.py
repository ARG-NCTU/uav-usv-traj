from setuptools import setup, find_packages

setup(
    name="hf_tools",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'huggingface_hub',
        'datasets'
    ],
    # Optional
    author="uwe",
    author_email="uwe90711@gmail.com",
    description="A tools for easier use of huggingface_hub",
    keywords="hf tool",
    url="https://github.com/ARG-NCTU/uav-usv-traj.git",
)

# ## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
# from distutils.core import setup
# from catkin_pkg.python_setup import generate_distutils_setup

# # fetch values from package.xml
# setup_args = generate_distutils_setup(
# 	packages=['hf_tools'],
# 	package_dir={'': 'src'},
# )
# setup(**setup_args)