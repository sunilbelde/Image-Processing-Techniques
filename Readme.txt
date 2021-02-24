FILE STRUCTURE:
===============
input -This Directory contains all the input images for the project to tun.

output - This Directory is initially empty and the create output images are copied here.

writeup.html - This html file has static page structure to display all the results.

Compiler.txt - This file contains about the installation and setup of the environment

requirements.txt - This file is used to install all the necessary packages for the project to run

src/ - This Directory contains the source code
       
	   Create.py - This python file is used to generate all the output images at once which has to show up in  the writeup.html page
	   Imagepro.py -This python file is used to generate output images one by one through command line.
	   
HOW TO PROCEED:
===============

1.Read compiler.text file and setup the environment.

2.In linux cd into the root directory of project.

3.Run the below command to generate all the output files at once:
  $ python src/Create.py (Linux)
  $ py src/Create.py (Windows)
  
  For pyhton commands use 'py' in windows and 'python' in Linux
  
  Once this is executed move to next step.
  
4.Open the writeup.html to see the results.

5.If you want to generate images one after other run the commands that are available in the writeup.html page.