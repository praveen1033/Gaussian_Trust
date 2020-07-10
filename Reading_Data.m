File_data = csvread('File6new.csv',1,1);                                      % Loads file to matrix called File_data
I = unique(File_data(:,1));                                                             % I includes all smart meter ids.
N = length(I);                                                                                 % N is the number of smart meters.
Meter_Readings = File_data(:,3)*1000;                                       % Converts the usage in kilowatts to watts.