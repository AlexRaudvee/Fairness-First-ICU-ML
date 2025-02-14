# Fairness-First-ICU-ML

### Set Up

1. **Create a Virtual Environment**
    To avoid dependency conflicts, create and activate a virtual environment:
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate it (Windows)
    venv\Scripts\activate  

    # Activate it (Mac/Linux)
    source venv/bin/activate  
    ```
    In our project we use python 3.10 due to its stability. 

2.  **Install Dependencies**
    Before running the project, install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
### Prepare to Data Download

1. Create a user account on [this](https://physionet.org) website
2. Read and sign the data use agreement on [this](https://physionet.org/sign-dua/widsdatathon2020/1.0.0/) website
3. Go to the `config.py` file and change your `USER_NAME` according to one that you entered in step 1.

### How to Execute

1. Make sure you did download data according to the instructions in [Data Download](#prepare-to-data-download) section
2. Run the data loader script to retrieve and set up the dataset:
    ```bash
    python dataloader.py
    ```

Now you're all set to further development! 🚀

