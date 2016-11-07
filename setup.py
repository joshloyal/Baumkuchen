from setuptools import setup


PACKAGES = [
        'baumkuchen',
        'baumkuchen.tests',
]

def setup_package():
    setup(
        name="baumkuchen",
        version='0.1.0',
        description='Repository for various Keras models',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/Baumkuchen',
        license='MIT',
        install_requires=['numpy', 'scipy', 'keras', 'tensorflow'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
