Analysis code for characterizing model states/outputs is in rnn_analysis_utils.py

Analysis code for characterizing model weights is in PongRNNWeights.py

Each RNN is characterized using PongRNNSummarizer, which parses individual RNN outputs and,
using the analysis libraries above, characterizes them.

PongRNNExpSummary pools RNN outputs with specific constraints,
selects specific RNNs (based on performance),
and compiles a summary of their characteristics.

This generates an output model_dat file, which contains everything needed to compare with primates.


PongRNNExpSummary >> PongRNNSummarizer >> (PongRNNWeights, rnn_analysis_utils)