# Track-Before-Detect
A torch based code for a ViterbiNet algorithm composed of a U-net architecthure and a Radar tracker consisting of: Hough transform tracker, single fram detection and viterbi

Files 'unet_model' and 'unet_parts' were taken from https://github.com/milesial/Pytorch-UNet
Subtle changes were made to accomodate our research

Data: The data was created using Matlab. The file 'readmatlabdata' reads data meant for training and validation. the file 'reattrackfrommatlab' reads tracks

The files 'Tracker' and 'tracker_parts' are the tracking algorithms
the files 'Train_Model' and 'train_parts' are for training
the file 'plot_parts' is for plotting
