import os

from setuptools import find_packages, setup


REQUIREMENTS_FILE = os.path.join(
    os.path.dirname(__file__), "lora_requirements.txt"
)


setup(
    name="lora_diffusion",
    py_modules=["lora_diffusion"],
    version="0.1.7",
    description="Low Rank Adaptation for Diffusion Models. Works with Stable Diffusion out-of-the-box.",
    author="Simo Ryu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lora_add = lora_diffusion.cli_lora_add:main",
            "lora_pti = lora_diffusion.cli_lora_pti:main",
            "lora_distill = lora_diffusion.cli_svd:main",
            "lora_ppim = lora_diffusion.preprocess_files:main",
        ],
    },
    install_requires=[
        line.strip()
        for line in open(REQUIREMENTS_FILE)
        if line.strip() and not line.lstrip().startswith("#")
    ],
    include_package_data=True,
)
