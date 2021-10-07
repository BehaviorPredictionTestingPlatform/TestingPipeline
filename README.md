# Behavior Prediction Testing Platform

## Setup

### Installing CARLA
* Download the latest release of CARLA. As of 10/6/20, this is located [here](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1)
    * Other releases can be found [here](https://github.com/carla-simulator/carla/releases)
    * First, download “CARLA_0.9.10.1.tar.gz”. Unzip the contents of this folder into a directory of your choice. In this setup guide, we’ll unzip it into “~/carla”
    * Download “AdditionalMaps_0.9.10.1.tar.gz”. Do not unzip this file. Rather, navigate to “~/carla” (the directory you unzipped CARLA into in the previous step), and place “AdditionalMaps_0.9.10.1.tar.gz” in the “Import” subdirectory.
* In the command line, cd into “~/carla” and run `./ImportAssets.sh`
* Try running `./CarlaUE4.sh -fps=15` from the “~/carla” directory. You should see a window pop up containing a 3D scene.
* The CARLA release contains a Python package for the API. To use this, you need to add the package to your terminal’s PYTHONPATH variable as follows:
    * First, copy down the filepath of the Python package. The package should be located in “~/carla/PythonAPI/carla/dist”. Its name should be something like “carla-0.9.10-py3.7-linux-x86_64.egg”
    * Open your “~/.bashrc” file in an editor. Create a new line with the following export statement: “export PYTHONPATH=/path/to/egg/file”
    * Save and exit “~/.bashrc” and restart the terminal for the changes to take effect. To confirm that the package is on the PYTHONPATH, try the command “echo $PYTHONPATH”

### Installing Scenic
* In a new terminal window, clone [this repository](https://github.com/BehaviorPredictionTestingPlatform/Scenic).
* Run `poetry install`

### Installing VerifAI
* In a new terminal window, clone [this repository](https://github.com/BehaviorPredictionTestingPlatform/VerifAI).
* In the command line, enter the repository and switch to the branch "kesav-v/multi-objective"
* Run `poetry install`

### Configuration Settings

This file lets you configure which Scenic programs to simulate, how many times to simulate each program, how many steps to run each simulation for, and where to output the generated data.

A sample configuration file, which must be saved in the JSON format, is shown below. Feel free to change the list of scripts to reference any Scenic programs on your machine.

```
{
   "scenic_programs": [
      "scenarios/ex.scenic"
   ],
   "behavior_prediction_model": "/path/to/model",
   "timepoint": 50,  // time step to begin prediction
   "sampler_type": "mab",  // VerifAI sampler
   "parallel_workers": 1,  // number of parallel workers
   "simulations_per_scenario": 3,  // number of iterations to execute each Scenic program
   "output_dir": "/path/to/output/dir",
   "time_per_simulation": 20,  // max time steps per simulation
   "input": [  // input data type(s)
      "trajectory"
   ],
   "headless_mode": false,  // render HUD in CARLA
}
```

## Running the Testing Pipeline

* Run `python main.py`
* This will save the error table CSVs to the output directory specified in the configuration file.