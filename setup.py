from setuptools import find_packages, setup


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


INSTALL_REQUIRES = [
    "scikit-learn>=1.0.1",
]

if __name__ == "__main__":
    setup(
        name="wtfml",
        description="WTFML: Well That's Fantastic Machine Learning",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Abhishek Thakur",
        author_email="abhishek4@gmail.com",
        url="https://github.com/abhishekkrthakur/wtfml",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        platforms=["linux", "unix"],
        python_requires=">=3.6",
        install_requires=INSTALL_REQUIRES,
    )
