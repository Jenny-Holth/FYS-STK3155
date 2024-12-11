# FYS-STK3155

This is the folder for Project 3 by Jenny Holth Hartting and Alexander Schei.

Our code for project 3 is split over "RunFile.ipynb", "Classes.py", "Functions.py" and "Utilities.py". 
"RunFile.ipynb" is a Jupyter Notebook that runs all the code found in "Classes.py" and "Functions.py". 
"Classes.py" contains the main class and a function for initializing the the neural network class. 
"Functions.py" contains all outside functions, and "Utilities.py" contains all small functions like cost and activation funtions, they are stored seperate to avoid import loops as both other .py files need to import them. 
In cases where our code is based on other work, it will be stated in the top of the ".py" file as well as above the relevant code.

The data folder contains all result data in .txt files, stored from previous runs. This data can be loaded in the RunFile and is recommended as some parts of the code take very long to run. Instructions on when and how to use stored data is explained in the RunFile notebook.

In the figures folder you will find all figures and plots from the "RunFile.ipynb", including some extra heatmaps, histograms and confusion matrices.

We hope you enjoy your stay in "Project 3" :)
