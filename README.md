# DBT_Diffusion: Denoising Diffusion Probabilistic Models for Digital Breast Tomosynthesis Augmentation

## Overview

DBT_Diffusion is a tool designed to augment Digital Breast Tomosynthesis (DBT) datasets using Denoising Diffusion Probabilistic Models. This project provides a powerful yet privacy-preserving solution for generating synthetic DBT samples, enabling improved training and generalization of machine learning models in medical imaging.

## Requirements

- Python 3.8
- Conda or venv for environment management

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/DBT_Diffusion.git
    ```

2. Create and activate a virtual environment:

    ```bash
    # Using conda
    conda create --name dbt_diffusion python=3.8
    conda activate dbt_diffusion
    
    # Using venv
    python -m venv dbt_diffusion
    source dbt_diffusion/bin/activate  # On Linux or macOS
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Ensure Dataset Format:**
    - Make sure your DBT dataset contains files in the `.dcm` (DICOM) format. This is essential for compatibility with the processing pipeline.

2. **Modify Dataset Path:**
    - Before running the training script, ensure to modify the path to your own DBT dataset. Open `multi_gpu_training.py` and locate the line specifying the dataset path:

    ```python
    folder = "/path/to/your/dbt/dataset"
    ```

    Replace `/path/to/your/dbt/dataset` with the actual path to your DBT dataset.

3. Launch the multi-GPU training script:

    ```bash
    torchrun --standalone --nnodes=<N_NODES> --nproc_per_node=<YOUR_GPUS> multi_gpu_training.py
    ```

    Ensure to replace `<N_NODES>` and `<YOUR_GPUS>` with the appropriate values for your setup.

4. Follow the on-screen instructions for additional configuration and parameter adjustments.

## Multi-GPU Support

DBT_Diffusion includes multi-GPU support for efficient training. Utilize the `torchrun` command with the specified parameters to distribute the workload across multiple GPUs.

## Contributions

Contributions to DBT_Diffusion are encouraged. If you discover bugs, have feature suggestions, or wish to contribute improvements, please create an issue or submit a pull request.

## License

DBT_Diffusion is licensed under the GNU GENERAL PUBLIC LICENSE. Refer to the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is intended for research and educational purposes. Adhere to ethical guidelines and legal regulations when handling medical data. Use DBT_Diffusion responsibly and comply with privacy and data protection laws.
