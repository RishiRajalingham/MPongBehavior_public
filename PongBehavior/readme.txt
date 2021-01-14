-- Each individual behavioral data file (MWorks to MATLAB) is processed using MentalPongBehavior.

-- A combination of files (e.g. across sessions) for one MODEL/SYSTEM is processed using BehavioralDataset.

-- Different MODELS/SYSTEMS are compared using BehavioralCharacterizer.
The BehavioralCharacterizer object is the core processing data structure.
-selects conditions that are common to all datasets
-computes metrics on each dataset using the exact same procedure
-computes comparison metrics (consistency) across datasets
-computes null distribution of metrics per dataset

-- The BehavioralCharacterizer is summarized with BehavioralComparer.
The main goal here is to parse data from many models and primates,
and accumulate consistency/performance metrics with other model attributes,
saving all in a single summary dataframe.

BehavioralComparer >> BehavioralCharacterizer >> BehavioralDataset >> MentalPongBehavior