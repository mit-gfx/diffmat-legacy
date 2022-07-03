<div align="middle">
  <img width="500px" src="misc/diffmat_logo.png">
</div>
<br/>

[![Latest Release](https://img.shields.io/badge/diffmat-0.0.1-blue)]()

## Important Notice
This repository contains the legacy version of DiffMat. The sole purpose of releasing this repository is to fulfill the agreement between Massachusetts Institute of Technology and Adobe Inc. All users are **not recommended** to use this repository. Please use the actively maintained new version hosted in [mit-gfx/diffmat](https://github.com/mit-gfx/diffmat).


## Overview

**DiffMat** is a differentiable procedural material modeling library built on [Pytorch](https://pytorch.org). It is designed to reproduce the functionality and match the output of [Adobe Substance Designer](https://www.substance3d.com/products/substance-designer/). DiffMat supports automatic conversion of SBS files (.sbs) into differentiable graphs.

## Requirements and Installation
DiffMat requires [Substance Automation Toolkit](https://www.substance3d.com/substance-automation-toolkit/) to produce input noises and patterns. For educational license users, please send an access request through [this page](https://www.substance3d.com/education/). For non-educational license users, please send an access request through [this page](https://www.substance3d.com/contact-enterprise/). By default, our parser loads Substance Automation Toolkit from Home directory for Linux and Mac, and Desktop for Windows. For custom install location, specify your toolkit path for the parser.

Once installed, download the DiffMat source and move into the source folder
```bash
git clone git_path_to_this_repo
cd diffmat
```
To setup a separate Conda environment with all dependencies:
```bash
conda env create -f environment.yml
conda activate diffmat
```
To install the package under the current python environment
```bash
pip install .
```
To install in development mode:
```bash
pip install -e .
```
Download freeimage plugin to load .exr format images
```bash
python -c "import imageio; imageio.plugins.freeimage.download()"
```


## Getting Started
The simplest and the most common way to use DiffMat is converting an existing .sbs file into a DiffMat graph. To do so, run the following command line command after installation:
```
sbs_parse PATH_TO_SBS_FILE
    -a: enable alpha channel
    -c: number of random seeds for generating noises
    -d: GPU device ID (default 0)
    -g: name of the output graph
    -k: add automatic normal correction to eliminate normal bias
    -l: lambertian BRDF, force the output roughness to be 1 and matallic to be 0
    -n: save noise images ('exr'|'png'|'jpg', default 'exr')
    -o: output file path (default to the current folder)
    -r: output resolution (default 9 (2^9=512))
    -t: substance automation toolkit path (default to home directory for Linux, Mac, and Desktop for Windows)
    -s: add a special transform at the end of graph to scale and offset the output
    -m: max scale of the special transform (default 4.0)
```
Running this command will generate a folder named identical to your graph at the specified output path. The folder is structured as follows
```
GRPAH_NAME
│
└───a set of useful .py scripts instantiated from diffmat/templates
│
└───random_seeds
│   │   a set of .exr images as inputs noises/patterns to the graph
│
└───configs
    │   a set of empty configuration files that users can modify to control the behavior of .py scripts in the root directory
```
For the functionality of each generated .py script, refer to [Code Structure](#code-structure). For writing a new template script, refer to [Write A Template Script](#write-a-template-script).

## Code Structure
```
diffmat
│
└───sbs
│   │   a collection of free SBS materials that DiffMat can convert
│
└───sbs_core
│   │   functional.py: implemention of SBS nodes.
│   │   nodes.py: node wrappers for graph construction.
│   │   util.py: utility functions.
│
└───sbs_convert
│   │   sbs_functions.py: function graph definitions for SBS parser.
│   │   sbs_nodes.py: node and parameter definitions for SBS parser.
│   │   sbs_parser.py: SBS file parser.
│
└───templates
    │   tp_eval.py: script to evaluate the graph with default parameters.
    │   tp_grid.py: script to generate a family of materials by sampling the graph parameters.
    │   tp_opt.py: script to optimize for a target photograph.
    │   tp_predictor.py: script to train a parameter prediction network.
    │   tp_stat.py: script to profile the runtime and memory cost of the graph.
    │   tp_util.py: graph definition translated from the input SBS file.

```

## Write A Template Script
A template script is a standard python script with a set of pre-defined macros to be replaced by graph-specfic outputs during the conversion. 

| Macro       | Replacement |
| ----------- | ----------- |
| GRAPH_NAME  | name of the sbs file |
| GRAPH_EXPOSED_PARAMS  | dictionary of the exposed parameters |
| GRAPH_DYNAMIC_FUNCTIONS | dynamic functions that map the exposed parameters to node parameters|
| GRAPH_TP_PARAMS | dictionary of the partial functions that apply the mapping from exposed parameters to node parameters  |
| GRAPH_PARAMS | dictionary of node parameters |
| NODE_LIST | list of nodes in program order |
| USE_ALPHA | use alpha flag |
| INPUT_NAMES | dictionary of input noise and noise file names |
| FORWARD_PROGRAM | forward evaluation program |
| CLASS_NAME | class name of the converted graph |
| FORWARD_CLASS | forward evaluation program in the graph class |
| NOISE_COUNT | number of copies per input noise map |


## Reproducing MATch Paper (SIGGRAPH Asia 2020)
- To evaluate a converted material with default parameters
```bash
cd GRAPH_NAME
python GRAPH_NAME_eval.py --no_ptb
```
Here ``--no_ptb`` stands for no perturbation. See other options provided by the script to control the evaluation process.

- To profile the runtime and memory cost of the graph
```bash
python GRAPH_NAME_stat.py
```

- To produce a grid of materials by sampling the graph parameters and save the sampled parameters
```bash
python GRAPH_NAME_grid.py
```
See other options provided by the script to control the sampling process and grid visualization.

- To train a parameter prediction model same as the one used in the paper
```bash
python GRAPH_NAME_predictor.py --fix_vgg
```
Please also configure the parameter sampling process (sampling method and perturbation strength) for your own applications. To define new models or modify the current model, install diffmat in develop mode and change ``diffmat/util/model.py``.

- To evaluate a trained model for target images
```bash
python GRAPH_NAME_predictor.py
```
with following configurations
```
--fix_vgg
--eval_mode
--load_epoch EPOCH_TO_BE_LOADED
--use_real_image
real_image_paths=[LIST OF TARGET IMAGE PATHS]
```

- To evaluate a trained model for an example in the grid (generated by GRAPH_NAME_grid.py)
```bash
python GRAPH_NAME_predictor.py
```
with following configurations
```
--fix_vgg
--eval_mode
--load_epoch EPOCH_TO_BE_LOADED
--load_grid_parameter
--grid_id GRID_ID
grid_example_ids=[LIST OF EXAMPLE IDs]
```

- To optimize a material for target images from default parameters
```bash
python GRAPH_NAME_opt.py
```
with following configurations
```
save_ids=[LIST OF IDs USED TO SAVE THE RESULTS]
--use_real_image
real_image_paths=[LIST OF TARGET IMAGE PATHS]
```


- To optimize a material for examples in the grid (generated by GRAPH_NAME_grid.py) from default parameters
```bash
python GRAPH_NAME_opt.py
```
with following configurations
```
save_ids=[LIST OF IDs USED TO SAVE THE RESULTS]
--load_grid_parameter
--grid_id GRID_ID
grid_example_ids=[LIST OF EXAMPLE IDs]
```

- To optimize a material for target images from network predictions
```bash
python GRAPH_NAME_opt.py
```
with following configurations
```
save_ids=[LIST OF IDs USED TO SAVE THE RESULTS]
--use_real_image
real_image_paths=[LIST OF TARGET IMAGE PATHS]
--load_network_prediction_real
net_prediction_real_image_postfix=[LIST OF TARGET IMAGE NAMES]
```

- To optimize a material for examples in the grid from network predictions
```bash
python GRAPH_NAME_opt.py
```
with following configurations
```
--load_network_prediction_syn
--net_prediction_grid_id GRID_ID
net_prediction_example_ids=[LIST OF EXAMPLE IDs]
```


## Citation
If you use DiffMat in your research and found it helpful, please consider citing the following paper:
```bib
@article{shi2020match,
author = {Shi, Liang and Li, Beichen and Ha\v{s}an, Milo\v{s} and Sunkavalli, Kalyan and Boubekeur, Tamy and Mech, Radomir and Matusik, Wojciech},
title = {MATch: Differentiable Material Graphs for Procedural Material Capture},
year = {2020},
publisher = {Association for Computing Machinery},
volume = {39},
number = {6},
issn = {0730-0301},
articleno = {196},
numpages = {15},
}
```

## License
DiffMat is released under a custom license from MIT and Adobe Inc. Please read our attached [license file](LICENSE) carefully before using the software. We emphasize that DiffMat **shall not** be used for any commercial purposes.