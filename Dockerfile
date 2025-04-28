# Use official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files into the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the default command to run the app
CMD ["python", "main.py"]

