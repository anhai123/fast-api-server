# Sets the working directory inside the container to /app
WORKDIR /app

# Copies all the contents from the current directory (which is src in this case) to the /app directory inside the container
COPY . /app

# Makes the entrypoint.sh script executable
RUN chmod +x entrypoint.sh

# Installs the required Python packages
RUN pip install -r requirements.txt

# Specifies the command to run when the container starts
CMD ["sh", "entrypoint.sh"]
