# Convolutional Neural Networks Seminar
This repository includes all relevant files used for the Convolutional Neural Networks Seminar

### Instructions:
1. In an Ubuntu machine, download/clone this repository.
2. Open a Terminal and execute the following commands to install the most recent version of python, pip3 and python virtualenv:  
`sudo apt update`  
`sudo apt install python3-dev python3-pip`  
`sudo pip3 install -U virtualenv`  
3. Create a new virtual environment, choosing a Python 3 interpreter ahd saving it in the `~/venv` directory, by executing the command:  
`virtualenv --system-site-packages -p python3 ~/venv`  
4. Activate the virtual environment using a shell-specific command:  
`source ~/venv/bin/activate`  
5. When the virtualenv is active, your shell prompt is prefixed with `(venv)`. Now it's possible to install packages within the virtual environment without affecting the host system setup. Start by upgrading pip:  
`pip install --upgrade pip`  
6. Install the TensorFlow pip package:  
`pip install --upgrade tensorflow`  
7. Verify the install:  
`python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`  
8. Install the matplotlib pip package:  
`pip install matplotlib`  
9. Install the python TKinter package:  
`sudo apt-get install python3-tk`  
10. Go to the Ubuntu Software app and install the __"Pinta"__ application.  
11. Run the desired TensorFlow scripts, and when finished exit the virtual environment session by running the command:  
`deactivate`
