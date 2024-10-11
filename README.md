This project is part of the manuscript entitled "Optimal feature selection and model explanation for reef fish sound classification" submitted to Philosophical Transactions B, as part of the special issue "Acoustic monitoring for tropical ecology and conservation".

The main goal of the manuscript is to classify unknown fish sounds using supervised algorithms. We used XAI to determine the contribution of acoustic features in
predicting each class. We also evaluated the effect of data augmentation  on the classifiers.

All files are stored in the 'Special Issue - PTB - Paper submission' folder.

We have provided some wav files with examples of the detected sound classes. The beginning and ending time of each fish call can be seen in the 'features_time.xlsx' file.

'feature_extraction.py' contains the code to extract acoustic features from 'features_time.xlsx file and wav files.
'supervised_classifiers.py' contains the code to run NB, DT, and RF classifiers using the files containing the feature sets
'MLP_final.py' contains the code to run MLP using the files containing the feature sets.
'cross_val_supervised' and 'cross_val_MLP' contains the code to perform cross-validation
'shap_plots.py' contains the code to compute and plot SHAP values and feature importance by class
'shap_polar.py' contains the code for thesummary plot and the polar plot of SHAP values for all classes of sounds.

The files for the main feature sets are: 'featureset1.xlsx', 'featureset2.xlsx', 'featureset3.xlsx'.
The files for the augmetend main feature sets are: 'featureset1_aug.xlsx', 'featureset2_aug.xlsx', and 'featureset3_aug.xlsx'.






