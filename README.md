# Mental Workload Detection Using IFMMoN

![Python version](https://img.shields.io/badge/Python-%3E3.7-orange)
![TensorFlow version](https://img.shields.io/badge/TensorFlow-2.X-orange)

## Overview of content

There are several relevant folders in the XDF folder: "Models", "Pipeline", and "TFRecord".
Furthermore, there are two files ([utils.py] & [get_et_data.py]) that contain various utility functions and a script to stream TobiiProFusion data to LSL, respectively.

### 1. [Models]

In general, all models follow the same principles: Each device has its own Base network
 All Bases feed into the same Head network, which converges the input data to a classification between zero and one. 

- [mlp_base.py]: DNN where both the Bases and the Head consist of only dense layers.
- [small_mlp_base.py]: The same network structure as [mlp_base.py], but with exactly half the number of units in each dense layer.
- [literature_base.py]: DNN where all Bases are based on what is commonly done in literature. 
The Head network consists of several dense and convolutional layers. 
- [small_literature_base.py]: The same network structure as [literature_base.py], but with exactly half the number of units and filters in each layer.

### 2. [Pipeline]

This folder contains mostly infrastructural and data-related scripts that are used for the creation of a dataset. 
Once a dataset is made, these scripts are no longer needed for training. 

- [xdf_import.py]: From this file, use `read_xdf()` to convert the recorded XDF file into a pandas.DataFrame (df) that contains all (meta)data. 
Please note that this dictionary should not be saved to a `.csv` file as some cells contain data in higher dimensionality.
Hence, the output of this function is only to provide a df that can be worked with in `create_chunk()`.
- [chunkify.py]: Use the `create_chunk()` function to convert your DataFrame from `create_data()` into `.csv` files, selected around markers. 
See script for options on file creation, such as epoch_length and output folder.

### 3. [TensorFlow]

Contains all scripts that are related to TensorFlow operations, such as TFRecord creation and training.

- [tfrecord_writer.py]: Writes TFRecord files that contain samples with the following information: device (meta)data, labels, andparticipant information.  
- [tfrecord_reader.py]: Reads TFRecord files that are created using the `TFRecordCreator` class from [tfrecord_writer.py].
- [trainer.py]: Defines the `Trainer` class that is used to train various networks on the created datasets.
- [hpo_search.py]: [Optuna] based script that searches hyperparameters and model combinations that minimises testing loss. 
Please note that doing this is computationally expensive. It is recommended to only do this if you have lots of compute available to you. 

---
Please do not hesitate to [send me an email] if you have any questions.

[Models]:src/Models
[mlp_base.py]:src/Models/mlp_base.py
[small_mlp_base.py]:src/Models/small_mlp_base.py
[literature_base.py]:src/Models/literature_base.py
[small_literature_base.py]:src/Models/small_literature_base.py

[Pipeline]:src/Pipeline
[chunkify.py]:src/Pipeline/chunkify.py
[xdf_import.py]:src/Pipeline/xdf_import.py

[TensorFlow]:src/TensorFlow
[hpo_search.py]:src/TensorFlow/hpo_search.py
[tfrecord_reader.py]:src/TensorFlow/tfrecord_reader.py
[tfrecord_writer.py]:src/TensorFlow/tfrecord_writer.py
[trainer.py]:src/TensorFlow/trainer.py

[utils.py]:src/utils.py
[get_et_data.py]:src/get_et_data.py
[Optuna]:https://optuna.org/
[send me an email]:mailto:t.c.dolmans@gmail.com
 