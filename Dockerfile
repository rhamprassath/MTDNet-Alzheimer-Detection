# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements_hf.txt
RUN pip install --no-cache-dir -r requirements_hf.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
