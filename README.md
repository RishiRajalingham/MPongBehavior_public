System requirements (software):
-python 3 (numpy, scipy, pickle, pandas, sklearn)

System requirements (hardware):
Disk space: 
    -pong_rnn dataset: ~570G
    -primate behavior (processed): ~5GB
    -pong_rnn model behavior (processed, individual): ~160G
    -pong_rnn model behavior (processed, summary): ~1.1G
    
Tested on:
-python 3.6
--numpy
--scipy
--pickle
--pandas
--sklearn

Installation instructions (<1hr on typical desktop computer):
-Python installation: https://realpython.com/installing-python/

Instructions:

-PongDatasets:
    Use PongDataset to create pong_rnn training and validation dataset (in raw pixel format, with labels). 
    Use DatasetEncoder to create sensory inputs modeled as pixels_pca and gabors_pca.

-PongRnn
    Use rnn_train/PongExperiment_prediction to train RNNs on datasets created in step 1. 
        All hyperparameter choices for different model sets are in the bash_drivers/ subdirectory, 
        which can be used to retrain all (hundreds of) models. 
    Use rnn_analysis/PongRNNExpSummary to analyze trained RNNs and create a processed RNN file.

-PongBehavior
    Use BehavioralDatasets to create/point to human and monkey behavior datasets.
    Use BehavioralCharacterizer/BehavioralCharacterizer to measure error metrics from primate/RNN behavior.
    Use BehavioralCharacterizer/BehavioralComparer to compare behavior across hundreds of RNNs and primates.

Figures (Human_behavior, Monkey_learning, RNN_dynamics, RNN_to_Primates, RNN_velocity_code) are generated using 
jupyter notebooks in tmp_notebooks. 
    Raw or minimally processed data may be required for some figures. 
    Sample processed data is provided in tmp_sample_data to generate some figures. 
