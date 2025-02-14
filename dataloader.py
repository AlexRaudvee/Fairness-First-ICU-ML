import subprocess

from config import USER_NAME

# Define the command
command = [
    "wget", "-r", "-N", "-c", "-np",
    "--user", f"{USER_NAME}", "--ask-password",
    "https://physionet.org/files/widsdatathon2020/1.0.0/"
]

# Execute the command
subprocess.run(command)
