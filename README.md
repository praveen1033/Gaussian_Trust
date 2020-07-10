# AMI-security
Contains codes (MATLAB and Python) for detecting data falsification in AMI / Smartgrid.

Data Description:
  The smart grid data is available in time intervals of 1 hour or 30 minutes per each smart meter.
  The power consumption is either in watts or kilowatts. We use watts. So, we need conversion if needed. 1 kilowatt = 1000 watts.
  The next few lines show the sample of the smart meter data set. To view the data sample clearly click Blame option on the top right corner.
  
      time                    id          usage         Generation          Grid
  01/01/2016 00:00:00         26          0.5271        -0.00126            0.52586
  01/01/2016 00:00:00         9052        0.3711        -0.00723            0.53711
  01/01/2016 01:00:00         26          0.4631        -0.00006            0.46231
  01/01/2016 01:00:00         9052        0.6712        -0.00785            0.64112

We need the first 3 columns for this project. 
The first column sows the time. The sample shows we have the data readings per each hour.
The second column shows the smart meter id number. This could be used for meter by meter seperation of data.
The third column is the usage recorded in kilowatts. We will convert it to watts for our project.
