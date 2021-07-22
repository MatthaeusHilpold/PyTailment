"""
$ {PROJECT_NAME} -- small_cli.py
===============================================================
Small cli application for generating csv files of all the outputs of the curtailment detection and analysis tool.

Author: Matthaeus Hilpold (matthaeus.hilpold@vattenfall.com)
        WO-ESA
        BA Wind

Copyright (c) Vattenfall AB 2021 ALL RIGHTS RESERVED.
"""

import PyTailment
import pickle
import pandas as pd
from IPython.display import display

if __name__ == "__main__":

    root = 'data/'
    eventfile = root + '2021_GDT_Warning_Events.pkl'
    curtailmentfile = root + '2021_GDT_curtailment.pkl'
    setpointfile = root + '2021_GDT_setpoint.pkl'
    windfile = root + '2021_GDT_wind.pkl'
    alarmfile = root + '2021_GDT_alarms.pkl'

    turbine_events, curtailment_reports, SCADA_wind, SCADA_setpoint = PyTailment.load_data(eventfile,
                                                                                curtailmentfile,
                                                                                setpointfile,
                                                                                windfile,
                                                                                alarmfile)
    additional_flags = True
    fill_missing_data = True
    fill_rated = False
    count_zero_as_curtailment = True
    difference_ratio = 0.1
    print('This is a small CLI for the PyTailment library')
    print('Do you have custom data paths?')
    selection = input('y/n: ')
    if selection == 'y':
        print('please enter the path strings')
        root = input('root folder: ')
        eventfile = root + input('eventfile: ')
        curtailmentfile = root + input('curtailmentfile: ')
        setpointfile = root + input('setpointfile: ')
        windfile = root + input('windfile: ')
        alarmfile = root + input('alarmfile: ')

    print('Do you want to configure the curtailment detection scenario? (Otherwise default values will be used)')
    selection = input('y/n: ')
    if selection == 'y':
        print("please type out True/False, yes/no or y/n")
        var = input("fill missing (NaN) values with 0? ")
        if var not in ['True', 'yes', 'y']:
            fill_missing_data = False
        var = input("fill missing (NaN) values with rated power? ")
        if var in ['True', 'yes', 'y']:
            fill_rated = True
        var = input("Should 0 be counted as curtailment? ")
        if var not in ['True', 'yes', 'y']:
            count_zero_as_curtailment = False
        print("controller induced curtailments tend to oscillate...")
        var = input("...how much difference should be counted as a new curtailment event? ")
        if var in ['True', 'yes', 'y']:
            difference_ratio = True

    filter_known_curtailment = True,
    filter_technician = True,
    minimum_duration = 5*3600
    print('Do you want to configure the event ranking scenario? (Otherwise default values will be used)')
    selection = input('y/n: ')
    if selection == 'y':
        print("please type out True/False, yes/no or y/n")
        var = input("filter known curtailment? ")
        if var not in ['True', 'yes', 'y']:
            filter_known_curtailment= False
        var = input("filter technician curtailment? ")
        if var in ['True', 'yes', 'y']:
            filter_technician = True
        var = input("minimum duration? ")
        if var in ['True', 'yes', 'y']:
            var = input("please enter a minimum duration in seconds: ")
            minimum_duration = var

    print('Which turbine(s) should be used for the MI evaluation? if only one, enter same number, otherwise range.')
    lower = int(input('enter first turbine number: '))-1
    upper = int(input('enter last turbine number: '))-1

    root = "outputs/"

    curtailments, timeseries = PyTailment.extract_curtailment_windows(SCADA_setpoint, additional_flags=True,
                                                                      fill_missing_data=fill_missing_data,
                                                                      count_zero_as_curtailment=count_zero_as_curtailment,
                                                                      missing_as_rated=fill_rated,
                                                                      SCADA_wind=SCADA_wind,
                                                                      curtailment_reports=curtailment_reports,
                                                                      turbine_events=turbine_events,
                                                                      difference_ratio=difference_ratio)

    with open(root+"curtailments.pkl", "wb") as f:
        pickle.dump(curtailments, f)

    with open(root+"timeseries.pkl", "wb") as f:
        pickle.dump(timeseries, f)

    duration_stats = PyTailment.compute_duration_coverage(curtailments, turbine_events,
                                                          filter_known_curtailment=filter_known_curtailment,
                                                          filter_technician=filter_technician,
                                                          minimum_duration=minimum_duration)


    ranked_list_duration = PyTailment.produce_event_ranking(duration_stats, turbine_events)

    ranked_list_duration.to_csv(root + "ranked_list_duration.csv", index=False)

    ranked_list_mi = PyTailment.daily_density_events_by_mutual_information(curtailments[lower:upper], turbine_events)


    ranked_list_mi.to_csv(root + "ranked_list_mi.csv")

    print("scenario executed, please find the outputs in the output folder")

    while True:
        var = input("display turbine curtailment detection output? y/n")
        if var == 'y':
            var = int(input("enter turbine number: y/n"))+1
            print(curtailments[var])
        var = input("display timeseries? y/n")
        if var == 'y':
            var = int(input("enter turbine number: y/n")) + 1
            print(timeseries[var])
        var = input("Exit? y/n")
        if var == 'y':
            break