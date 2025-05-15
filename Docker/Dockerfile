# Use a Python base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy everything into the container
COPY . .

# Install required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Set the default command to run Streamlit
CMD ["streamlit", "run", "walmart2.py"]
