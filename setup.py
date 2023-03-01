from setuptools import setup

setup(
    name="mmpush",
    version="0.1",
    description="Tools and simulations for planar robotic pushing.",
    author="Adam Heins",
    author_email="mail@adamheins.com",
    install_requires=["numpy", "matplotlib", "scipy", "osqp", "pymunk"],
    packages=["mmpush"],
    python_requires=">=3",
    zip_safe=False,
)
