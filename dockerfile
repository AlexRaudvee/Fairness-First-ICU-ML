# Use an official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /Fairness-First-ICU-ML

# Copy all project files to the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run dataloader.py and then main.py sequentially
CMD ["sh", "-c", "python dataloader.py && python main.py"]