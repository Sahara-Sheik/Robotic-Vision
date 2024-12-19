"""
This file contains settings for the BerryPicker project. These can be considered as constants for a particular run, but they might have different values on different computers or setups.

They should be accessed through Config().values["value"]

A configuration system that allows porting between different machines and the use of multiple configuration files. 

It starts by loading a file from ~/.config/BerryPicker/mainsettings.yaml" 
from where the "configpath" property points to the path of the 
actual configuration file. 

Template configuration files FIXME and mainsettings-sample.yaml are in top directory of the project. However, the actual configuration files should be on the local directory outside the github package, as these configuration files contain local information such as user directory names etc.

The configuration values are a hierarchical dictionary which can be 
accessed  should be accessed through Config()["value"] or 
Config()["group"]["value"]


"""

import yaml
from pathlib import Path

class Config:
    """The overall settings class. """
    _instance = None  # Class-level attribute to store the instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            home_directory = Path.home()
            main_config = Path(home_directory, ".config", "BerryPicker","mainsettings.yaml")
            print(f"Loading pointer config file: {main_config}")
            if not main_config.exists():
                raise Exception(f"Missing pointer config file: {main_config}")
            with main_config.open("rt") as handle:
                main_config = yaml.safe_load(handle)
            configpath = main_config["configpath"]
            print(f"Loading machine-specific config file: {configpath}", flush=True)
            configpath = Path(configpath)
            if not configpath.exists():
                raise Exception(f"Missing machine-specific config file: {configpath}")
            with configpath.open("rt") as handle:
                cls._instance.values = yaml.safe_load(handle)
        return cls._instance

    def __getitem__(self, key):
        return self.values[key]
    
    def get_experiment(self, name):
        """Returns an experiment configuration, which is the 
        mixture between the system-dependent configuration and the system independent configuration."""
        # Load the system independent configuration
        current_directory = Path(__file__).resolve().parent
        experiment_sys_indep = Path(current_directory, "experiment_configs", name + ".yaml")
        if not experiment_sys_indep.exists():
            raise Exception(f"Missing experiment system independent config file {experiment_sys_indep}")
        with experiment_sys_indep.open("rt") as handle:
            indep_config = yaml.safe_load(handle)
        # Load the system dependent configuration
        experiment_directory = Path(self.values["experiment_system_dependent_dir"])
        experiment_sys_dep = Path(experiment_directory, name + "_sysdep.yaml")
        if not experiment_sys_dep.exists():
            raise Exception(f"Missing experiment system dependent config file {experiment_sys_dep}")
        with experiment_sys_dep.open("rt") as handle:
            dep_config = yaml.safe_load(handle)
        exp_config = indep_config | dep_config
        return exp_config
