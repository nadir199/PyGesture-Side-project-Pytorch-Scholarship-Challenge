# PyGesture-Side-project-Pytorch-Scholarship-Challenge
This is a python app associated with a flask web app for associating gestures to certain actions on your PC
# How to use it ?
## What to install ?
You need to have :
* **Python 3.6** or Higher installed
* **torch** installed.

## Python libraries
```
pip install flask
pip install keyboard
```

## How to run the program
* Go to the environment where you installed all the dependencies (Maybe Anaconda prompt)
* Go to the command line and run the command according to your OS :
  * Linux or Mac run : `export FLASK_APP=index.py`
  * Windows : `set FLASK_APP=index.py`
* Run this command in the command line : `flask run`
* You will get a link in the command line, generally http://127.0.0.1:5000 , Open it in your browser.
* Here, in this web app you can set the actions you want for each of the listed signs.
* There are 2 available gestures : 
  * Fist
  * High five
* There are 3 available types of actions : 
  * Opening a website
  * Pressing a key
  * Typing a phrase
* Choose the actions according to what you want ! Make sure to type URLs correctly.
  * If you choose Pressing a key, i recommend testing with "space" on a video ;) (Pause/Unpause with a gesture)
* Click on `Save Settings`
* Close the web app

* The following program will need to access your webcam, you can go over the code and see that there is no harm on activating your webcam :p.
* In order for the program to be able to access your webcam, you need to authorize it or suspend your antivirus T_T.
* Go to the command line and run :`python program.py`
* The program is now running ! Try one of the gestures facing the camera to trigger the actions ! :) 
* Have fun !

## How do I trigger actions ?
* Just make one of the gestures in front of the camera once the program runs :)
* Try not to be in a dark room.
* Don't stay too far from the camera, stay at a natural position.
