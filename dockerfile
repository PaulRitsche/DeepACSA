# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    tk-dev

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose port for running the app
EXPOSE 5000

# Define the command to run the app
CMD ["python", "./Deep_ACSA/deep_acsa_gui.py"]
