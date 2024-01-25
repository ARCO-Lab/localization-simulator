# Localization Simulator

## Description

A 2D/3D simulator for visualizing and performing experiments on trajectories in an environment to determine optimal sensor placement.

More changes to be made for the simulator and documentation!

## Commands

### Running the simulator
* `./run.sh` - Run the main simulation (2D). Make sure that the script is executable.
* `python -m localization_simulator.core.main --3` - Run a 3D simulation (currently not fully implemented).

### Viewing the documentation
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.

### Installing the conda environment/requirements
* `conda env create -f environment.yml` - Create the localization conda environment. Please note that you'll need to add pip as a dependency in the `environment.yml` file. You can still proceed without doing so but risk unwanted behavior.
Please change the prefix in the `environment.yml` to wherever you want to store the environment.
* `pip install -r requirements.txt` - Install all the packages specified in the requirements file into your own environment.

## Project layout

    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
    localization_simulator/
        core/ # Main simulator modules
            component.py
            error.py
            main.py # The file to run the simulation
            map.py
            plot.py
        site/ # Built documentation
        test/ # Where testing modules will be added
        utils/ # Utility modules
            helper.py
            nls.py
            trajectory.py
    environment.yml
    mkdocs.yml    # The configuration file.
    requirements.txt
    run.sh