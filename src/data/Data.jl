"""
    Data

Module for defining data types.

This module contains structs handling data, distributions, and data loaders. 
"""

module Data

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.DEFAULT_DEVICE as DEFAULT_DEVICE

import SciGenML.Config as Config

import Distributions
import JLD2
import Zarr

### Kolmogorov data loading ###
include("kolmogorov.jl")

export load_kolmogorov_data

function load_data(config)
    if config.data isa Config.KolmogorovDataHyperparameters
        return load_kolmogorov_data(config.data)
    end
end

export load_data

end
