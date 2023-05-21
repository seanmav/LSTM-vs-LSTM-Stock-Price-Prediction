In this system folder. 

You first have to install Anaconda.

From anaconda, launce the anaconda prompt.

Create a new conda environment.
Activate the conda environment.

Install the necessary modules, Keras, Pytorch, pandas, matplotlib
Install python 3 into the conda environment.
Install spyder on the conda environment

Launch spyder through the newly installed conda environment



Launch the file hybrid_lstm. (Main Run Script)

Change the file input at line 27.

Edit the train_size and test_size at line 23.

Everytime a new model is created it's base weight and error weight are stored in a file so the model doesn't have to be trained again.

Currently there are several base weights and error weights used.

The h6 is the model used in the report. Any other h# models are used to test and learn about epoch, batch_size.

Please read the Microsoft notepad to see the which base weight and error weight are associated with it's 
1.epoch
2.batch_size
3.train_size and test_size


The reverseCSV.py file is used to reverse the row orders of the excel sheet. However this python file can only be used with two columns of data.