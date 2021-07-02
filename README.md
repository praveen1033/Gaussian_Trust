# AMI-security
Contains codes (MATLAB and Python) for detecting data falsification in AMI / Smartgrid.

PAPER NAME: 

DATASET SOURCE: 

DATA DESCRIPTION: Pecan Street Texas Dataset
  The smart grid data is available in time intervals of 1 hour. 
  The power consumption in the data is in kilowatts (kW) and 1 kilowatt = 1000 watts.
  A sample of the smart meter data set is shown below.
  
| time                | id   | usage  | generation | grid    |
|---------------------|------|--------|------------|---------|
| 01/01/2016 00:00:00 | 26   | 0.5271 | -0.00126   | 0.52586 |
| 01/01/2016 00:00:00 | 9052 | 0.3711 | -0.00723   | 0.53711 |
| 01/01/2016 01:00:00 | 26   | 0.4631 | -0.00006   | 0.46231 |

We need the first 3 columns for this paper. 
The first column shows the time. The sample shows we have the data readings per each hour.
The second column shows the smart meter id number. This could be used for meter by meter seperation of data.
The third column is the usage recorded in kilowatts. We will convert it to watts for our project.

We will start with how to load the datafile. First, we will see how to load the file using MATLAB.

To load the csv file to matrix (we named File_data), use the following command
File_data = csvread('File6new.csv',1,1);                                      % Loads file to matrix called File_data

To extract the vector of list of smart meters, which is the list of smart meter ids
I = unique(File_data(:,2));                                                   % I includes all smart meter ids.

To get the number of smart meters in the dataset, we have to find the length of the vector for unique smart meter ids
N = length(I);                                                                % N is the number of smart meters.

To convert the data into watts, we have to multiply each reading by 1000
Meter_Readings = File_data(:,3)*1000;                                         % Converts the usage in kilowatts to watts.

Now, we will see how to clean the data.

WHY FILTERING NECESSARY ?


FILTERING LOGIC? 


