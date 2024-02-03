# Use the official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install required libraries for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV API_TOKEN="your_token_here"

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]