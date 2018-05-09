from setuptools import setup, find_packages

setup(
    name="MutationTestDNN",
    version="0.1",
    packages=find_packages(),
    # scripts=['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=['python=2.7.12, Tensorflow=1.3.0, keras=2.0.8'],

    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     '': ['*.txt', '*.rst'],
    #     # And include any *.msg files found in the 'hello' package, too:
    #     'hello': ['*.msg'],
    # },

    # metadata for upload to PyPI
    author="Jingyi Wang, Peixin Zhang",
    author_email="wangjyee@gmail.com",
    description="This is a project for mutation testing of deep neural networks",
    license="PSF",
    keywords="deep neural networks, mutation testing",
    # url="http://example.com/HelloWorld/",   # project home page, if any
    # project_urls={
    #     "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #     "Documentation": "https://docs.example.com/HelloWorld/",
    #     "Source Code": "https://code.example.com/HelloWorld/",
    # }

    # could also include long_description, download_url, classifiers, etc.
)