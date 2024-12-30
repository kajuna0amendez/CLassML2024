# Installation

1. Install docker for linux versions
+ instlal the toolkit if you have a nvidia support
+ "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
+ then do a "sudo apt-get install -y nvidia-container-toolkit"
+ Restart the docker "sudo systemctl restart docker"
2. Pull the following verion of jax 
+ docker pull ghcr.io/nvidia/jax:upstream-t5x-2024-12-27
3. Run the new docker container 
+ docker run --gpus all --net=host -it  --name jax_ml  ***id of image***  /bin/bash
+ NOTES: --gpus all if you have a gpu, --it is the interactivity for the bash
4. Use VSCODE with pylinter, dev container or whatever you want.
5. Inside the docker add 

# Project Structure
```
|-- backend
|   |-- core
|   |-- models
|   |   |-- linear_regression
|   |-- utils
|-- config
|-- examples
|-- frontend
|   |-- core
|-- tests
```

# How to use the API
