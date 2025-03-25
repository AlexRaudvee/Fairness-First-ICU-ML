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
4. If you encounter problems with downloading the data through our dataloader, you can do it manually. But then make sure that the data has the following path `physionet.org/files/widsdatathon2020/1.0.0/data` and inside this data derictory you should have `training_v2.csv` and `WiDS_Datathon_2020_Dictionary.csv`.

### How to Execute Locally

1. Make sure you did download data according to the instructions in [Data Download](#prepare-to-data-download) section
2. Run the data loader script to retrieve and set up the dataset:
    ```bash
    python dataloader.py
    ```
3. Then run the `main.py` file to execute and run the whole pipeline code.
4. If you encounter problems with some libraries, there is possibility that you have to adjust them manually in `requirements.txt` file. Otherwise try to run with Docker Container.

Now you're all set to further development! 🚀

### How to Run with Docker
Advantage of running with docker is that you can forget about systems dependencies, with docker it doesn't matter which OS you are using. Therefore if code doesn't run for you because of mysterious errors, you can try to run this code with Docker.

1. Make sure you did steps in [data preparation](#prepare-to-data-download) and did clone this repo.
2. Make sure Docker is installed on your system. You can download it from [Docker’s official website](https://docs.docker.com/desktop/)
3. Build the docker image: 
    ```bash 
    docker build -t fairness-first .
    ```
4. Run the container:
    ```bash 
    docker run -it my-container /bin/bash
    ```
    When you started to run the container, you can stop the session by typing ```exit``` in the terminal or press `Ctrl + D`.