## MultiNet Toolkit
The MultiNet Toolkit is a dashboard built for [IU Network Science Institute](https://iuni.iu.edu/) to facilitate science of science exploration and analyses.  Its intention is to be user friendly for institutional researchers on a global level.  blah blahblah.

## User Guide

The MultiNet Toolkit Userguide: everything that you need to know to be productive with the dashboard.

### Running the app locally

If you do not have python 3 installed, you need to install before continuing.  

Here is an easy way to get the latest install: [https://repo.anaconda.com/archive/]


Clone the github repository and navigate to dashboard_toolkit folder:
  ```
   git clone https://github.com/rebecca-my/multinet_dashboard
   
  ```
Navigate into multinet_dashboard folder:
  ```
  cd multinet_dashboard
  
  ```
Create a virtual environment:
```
  conda create -n dash_app_env --file requirements.txt -c conda-forge
  
```
Activate virtual environment:
```
  conda activate dash_app_env
  
```
Run the python file and you are all set!
```
  python multinet_toolkit.py
  
```
