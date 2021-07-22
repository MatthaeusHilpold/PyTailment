"""
$ {PROJECT_NAME} -- PyTailment.py
===============================================================
library functions to detect and analyze curtalment

Author: Matthaeus Hilpold (matthaeus.hilpold@vattenfall.com)
        WO-ESA
        BA Wind

Copyright (c) Vattenfall AB 2021 ALL RIGHTS RESERVED.
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from IPython.display import display, clear_output, HTML
from sklearn.feature_selection import mutual_info_classif
from IPython.display import display

def load_data(eventfile: str, curtailmentfile: str, setpointfile: str, windfile: str, alarmfile: str):
    """
    Load data dataframes from paths and add/rename columns as necessary.

    Args:
        [...]file : string path to the pickle file that contains the data loaded from azure.

    Returns:
        pandas DataFrames ready for processing.
    """
    with open(setpointfile, "rb") as fh:
        SCADA_setpoint = pickle.load(fh)

    with open(eventfile, "rb") as fh:
        turbine_events = pickle.load(fh)

    with open(curtailmentfile, "rb") as fh:
        curtailment_reports = pickle.load(fh)

    with open(windfile, "rb") as fh:
        SCADA_wind = pickle.load(fh)

    with open(alarmfile, "rb") as fh:
        turbine_alarms = pickle.load(fh)

    if 'Querydate' in curtailment_reports:
        curtailment_reports = curtailment_reports.drop(
            ['Querydate', 'From_Local', 'To_Local', 'Duration', 'Comment', 'Id', 'LostProduction', 'Park', 'Production',
             'ProductionLostTotal', 'ProductionTotal'], axis=1)
    if 'From_UTC' in curtailment_reports:
        # Rounding to closest 10 mins is a possible source of model bias
        curtailment_reports['From'] = curtailment_reports['From_UTC'].dt.floor('10min')
        curtailment_reports['To'] = curtailment_reports['To_UTC'].dt.round('10min')
        curtailment_reports = curtailment_reports.drop(['From_UTC', 'To_UTC'], axis=1)

    turbine_alarms['ErrorType'] = 'Alarm'
    if 'Unit' in turbine_alarms:
        turbine_alarms['Turbine'] = turbine_alarms['Unit']
        turbine_alarms = turbine_alarms.drop(['Unit'], axis=1)
    if 'From_UTC' in turbine_alarms:
        turbine_alarms['From'] = turbine_alarms['From_UTC'].dt.floor(
            '10min')  # possible cause for missed correlations if curtailment triggers shortly after event
        turbine_alarms['To'] = turbine_alarms['To_UTC'].dt.round(
            '10min')  # round should alleviate the above-stated problem
        turbine_alarms = turbine_alarms.drop(['From_UTC', 'To_UTC'], axis=1)

    if 'AlarmTxt' in turbine_alarms:
        turbine_alarms['Text'] = turbine_alarms['AlarmTxt']
        turbine_alarms = turbine_alarms.drop(['AlarmTxt'], axis=1)

    if 'AlarmCode' in turbine_alarms:
        turbine_alarms['ErrorCode'] = turbine_alarms['AlarmCode']
        turbine_alarms = turbine_alarms.drop(['AlarmCode'], axis=1)

    if 'Unit' in turbine_events:
        turbine_events['Turbine'] = turbine_events['Unit']
        turbine_events = turbine_events.drop(['Unit'], axis=1)
    if 'From_Floor' in turbine_events:
        turbine_events['From'] = turbine_events['From_Floor']
        turbine_events['To'] = turbine_events['To_Floor']
        turbine_events = turbine_events.drop(['From_Floor', 'To_Floor'], axis=1)
    if 'From_UTC' in turbine_events:
        turbine_events['From'] = turbine_events['From_UTC'].dt.floor(
            '10min')  # possible cause for missed correlations if curtailment triggers shortly after event
        turbine_events['To'] = turbine_events['To_UTC'].dt.round(
            '10min')  # round should alleviate the above-stated problem
        turbine_events = turbine_events.drop(['From_UTC', 'To_UTC'], axis=1)
    if 'ID' in turbine_events:
        turbine_events = turbine_events.drop(
            ['ID', 'Source', 'CustomText', 'Wallclock_UTC', 'CommFailure', 'MayTrigger', 'RemoteID', 'Created_UTC',
             'ErrorListID', 'RowLastUpdated'], axis=1)
    turbine_events = pd.concat([turbine_events, turbine_alarms],
                               ignore_index=True)  # merge alarms and events into one dataframe

    long_events = (turbine_events['To'] - turbine_events['From'] < pd.Timedelta(value=31, unit='days'))
    turbine_events = turbine_events[long_events]

    turbine_events['Duration'] = turbine_events.To - turbine_events.From

    return turbine_events, curtailment_reports, SCADA_wind, SCADA_setpoint


def within_margin(target_value, comparison_value, maxval: int, difference_ratio=0.02):
    """
    Evaluate whether the target value falls into a interval around the
    comparison value. The interval is defined as percentage of the maximum value
    for the dataset. This is used to determine oscillating values pertaining to
    the same curtailment event (controller curtailment) and gradual change
    as opposed to constant values (technician curtailment).

    Args:
        target_value: the value or vector of values being compared/checked
        comparison value: the value or vector of values
        at the center of the interval
        maxval: the highest value in the dataset
        difference_ratio: the ratio that defines the interval

    Returns:
        True if target_value is close enough to comparison_value. False othewise.
    """

    return np.abs(target_value - comparison_value) < (maxval * difference_ratio)

def during(start1: pd.Timestamp, end1: pd.Timestamp, end2: pd.Timestamp):
    """Check whether event 1 is happening during event 2
    (check for positive overlap of the two events)

    Args:
        start1: Timestamp indicating the start of event 1
        end1: Timestamp indicating the end of event 1
        end2: Timestamp indicating the end of event 2

    Returns:
        True if event 2 happens (partly) during event 1, False otherwise

    """

    latest = max(start1, end2)
    earliest = min(end1, end2)
    delta = pd.Timedelta((earliest - latest)).total_seconds()

    if delta < 0:
        return False
    return True


def expand_dummy_event_columns(turbine_curtailments: pd.DataFrame, turbine_events: pd.DataFrame, missing_as_na=False):
    """
    Add a dummy column for each event code to turbine curtailment containing the
    overlap ratio. 0 for non-ovelapping.

    Args:
        turbine_curtailments:  DataFrame containing detected curtailment events
        per turbine.
        turbine_events: DataFrame containing events and alarms for one turbine.
        missing_as_na: if True, exapnded dataframe will be filled with na's,
        otherwise with zeroes

    Returns:
        Adds a column for each event code and returns modified dataframe
    """

    filler = np.nan if missing_as_na else 0.
    event_codes = turbine_events.ErrorCode.unique()

    def create_list(event_codes):
        for code in event_codes:
            yield (str(code) + '_Curtailment_Coverage')
            yield (str(code) + '_Coverage_Duration')
            yield (str(code) + '_Coverage_Ratio')

    if isinstance(turbine_curtailments, list):
        for curtailment in range(len(turbine_curtailments)):
            # Need to generate list in every loop iteration, otherwise it "runs out
            # of objects"
            col_list = create_list(event_codes)

            extra_cols = pd.DataFrame(filler, index=turbine_curtailments[curtailment].index, columns=list(col_list))

            turbine_curtailments[curtailment] = pd.concat([turbine_curtailments[curtailment], extra_cols], axis=1)

    elif isinstance(turbine_curtailments, pd.DataFrame):

        col_list = create_list(event_codes)

        extra_cols = pd.DataFrame(filler, index=turbine_curtailments.index, columns=list(col_list))

        turbine_curtailments = pd.concat([turbine_curtailments, extra_cols], axis=1)
    else:
        print("Error: object to expand is neither a list of curtailments nor a single turbine curtailment dataframe!")

    return turbine_curtailments


def efficient_event_overlap_calculation(turbine_curtailments: pd.DataFrame, single_turbine_events: pd.DataFrame):
    # TODO: investigate and fix possible bug when multiple events with the same
    # code overlap a zero-duration curtailment window.

    """
    Evaluate overlap ratios between events and detected curtailment. 0 means no
    overlap.

    Same caveats as with efficient_curtailment_overlap_calculation apply.

    Args:
        turbine_curtailments:  DataFrame containing detected curtailment events
        per turbine.
        turbine_events: DataFrame containing events and alarms for one turbine.

    Returns:
        Void. Adds three columns for each event code containing the ratio of overlap
        of the event by curtailment, the ratio of overlap of the curtailment by the
        event and the duration of this overlap.

        NB: In previous versions of this project both the ratio of overlap for the
        event by the curtailment and for the curtailment by the event were
        determined. There is a case to be made that the latter might be an
        interesting metric too: currently, a very short event that happens during
        curtailment will have a high overlap ratio, but explain very little about
        the curtailment event.

        Deprecated:
          --Adds two columns: list of overlapping events and list of ratios.--
          Now replaced by dummy columns
    """

    if turbine_curtailments.columns.str.contains(pat='Coverage').any():

        for detected_turbine_curtailment in turbine_curtailments.iloc:
            overlappers = single_turbine_events[(~(single_turbine_events.index > detected_turbine_curtailment.End) & ~(
                        single_turbine_events.To < detected_turbine_curtailment.name))]
            for event in overlappers.iloc:
                code = str(event['ErrorCode'])
                coverage = pd.Timedelta(
                    min(detected_turbine_curtailment.End, event.To) - max(detected_turbine_curtailment.name,
                                                                          event.name)).total_seconds()
                curtailment_duration = pd.Timedelta(
                    detected_turbine_curtailment.End - detected_turbine_curtailment.name).total_seconds()
                if curtailment_duration == 0:
                    turbine_curtailments.at[detected_turbine_curtailment.name, code + '_Curtailment_Coverage'] = 1.
                else:
                    turbine_curtailments.at[detected_turbine_curtailment.name, code + '_Curtailment_Coverage'] = (
                                coverage / curtailment_duration)

                turbine_curtailments.at[detected_turbine_curtailment.name, code + '_Coverage_Duration'] += coverage

                event_duration = pd.Timedelta(event.To - event.name).total_seconds()
                if event_duration == 0:
                    turbine_curtailments.at[detected_turbine_curtailment.name, code + '_Coverage_Ratio'] = 1
                else:
                    turbine_curtailments.at[
                        detected_turbine_curtailment.name, code + '_Coverage_Ratio'] = coverage / event_duration

                """
                if code == '5112' and event_duration != 0 and curtailment_duration != 0:
                  display(detected_turbine_curtailment)
                  display(overlappers)
                  display(overlappers[overlappers.ErrorCode == 5112])
                  print("coverage ", coverage)
                  print("curtailment cov ", coverage/curtailment_duration)
                  print("coverage ratio> ", coverage/event_duration)
                  display("Curtailment coverage in output dataframe: ", turbine_curtailments.at[detected_turbine_curtailment.name, code + '_Curtailment_Coverage'])
                """

    else:
        print("please expand the columns first")

"""
def delete_long_events(turbine_events: pd.DataFrame, turbine_curtailments: pd.DataFrame):
    """"""
    Deletes all events in turbine_events that are longer than the longest curtailment

    Args:
        turbine_curtailments:  DataFrame containing detected curtailment events
        per turbine.
        turbine_events: DataFrame containing events and alarms for one turbine.

    Returns:
        trimmed DataFrame.
    """"""

    names = SCADA_setpoint["Turbine"].unique()
    names = np.sort(names)
    len_before = len(turbine_events)
    for i in range(1):
        a[i]['Duration'] = a[i].End - a[i].index
        turbine_events = turbine_events[
            ((turbine_events.To - turbine_events.From < a[i]['Duration'].max()) | (turbine_events.Turbine == names[i]))]
    print('Deleted ', len_before - len(turbine_events), ' long duration events from turbine event DataFrame')
    return turbine_events
"""

def high_wind_ride_through(current_turbine: pd.DataFrame, current_wind: pd.DataFrame):
    """
    Calculate whether for a certain timestamp high wind ride through is likely

    Args:
        current_turbine: DataFrame containing flagged data of one turbine
        current_wind: DataFrame containing SCADA ambient wind data for one
        turbine

    Returns:
        DataFrame with column with wind flags
    """

    current_wind = current_wind.set_index(['Timestamp'])
    current_wind = current_wind.sort_index()

    current_turbine['HWRT'] = (current_wind['Value'] > 24)

    current_turbine = current_turbine.fillna(value={'HWRT': False})
    return current_turbine


def technician_curtailment(current_turbine: pd.DataFrame):
    """
    Evaluate whether a curtailment is technician curtailment

    Args:
        current_turbine: DataFrame containing flagged data of one turbine
        current_wind: DataFrame containing SCADA ambient wind data for one
        turbine

    Returns:
        DataFrame with column with wind flags
    """

    previous_val = current_turbine['Value'].shift()
    next_val = current_turbine['Value'].shift(-1)
    after_next_val = current_turbine['Value'].shift(-1).shift(-1)

    # Technician curtailment is active if the turbine is curtailed and the next two
    # values are the exactly the same, or the previous one is exactly the same, as
    # the current one
    current_turbine['isTechnicianCurtailment'] = \
        current_turbine['isCurtailed'] & (((next_val == current_turbine['Value']) & \
                                           (after_next_val == current_turbine['Value'])) | \
                                          (previous_val == current_turbine['Value']))

    return current_turbine


def efficient_curtailment_overlap_calculation(turbine_curtailments: pd.DataFrame, curtailment_reports: pd.DataFrame,
                                              timeseries=False):
    """
    Compare discovered curtailments with curtailment reports and add a column
    containing a flag for whether there is overlap.

    Overlap is calculated optimistically in favor of reported curtailments
    if working with floored timestamps (as is the case with the default data
    loading for this project). For example, a detected curtailment might end at
    'xx-xx-xxxx 15-43-000' and a curtailment report might start at
    'xx-xx-xxxx 15-47-000'; subsequently, both are floored to the closest 10-minute
    interval at 'xx-xx-xxxx 15-40-000' and counted as overlapping.

    Args:
        turbine_curtailments:  DataFrame containing detected curtailment events
        per turbine
        curtailment_reports: DataFrame containing curtailment reports from
        fact.CurtailmentNew
        timeseries: Boolean. If True, will set end time to start time for time series.

    Returns:
        Void. Adds boolean column 'Reported' to turbine_curtailments.
    """

    curtailment_reports = curtailment_reports.sort_values(by=['From'])
    curtailment_reports = curtailment_reports.set_index('From')

    iterator1 = turbine_curtailments.iterrows()
    iterator2 = curtailment_reports.iterrows()
    index1, row1 = next(iterator1)
    index2, row2 = next(iterator2)
    turbine_curtailments['Reported'] = False

    # Only one iteration
    while True:
        try:
            if timeseries:
                range1 = pd.Interval(left=row1.name, right=row1.name)
            else:
                range1 = pd.Interval(left=row1.name, right=row1.End)
            range2 = pd.Interval(left=row2.name, right=row2.To)
            if range2.right < range1.left:
                # no overlap. range2 before r1. advance iterator2
                index2, row2 = next(iterator2)
            elif range1.right < range2.left:
                # no overlap. range1 before r2. advance iterator1
                index1, row1 = next(iterator1)
            else:
                # overlap. overlap(row1, row2) must > 0
                turbine_curtailments.loc[index1, 'Reported'] = True
                # determine whether to advance iterator1 or it2
                if range1.right < range2.right:
                    # advance iterator1
                    index1, row1 = next(iterator1)
                else:
                    # advance iterator2
                    index2, row2 = next(iterator2)
        except StopIteration:
            break


def extract_curtailment_windows(SCADA_setpoint: pd.DataFrame,
                                difference_ratio=0.1,
                                count_zero_as_curtailment=False,
                                additional_flags=False,
                                fill_missing_data=False,
                                missing_as_rated=False,
                                SCADA_wind=None,
                                curtailment_reports=None,
                                turbine_events=None):
    """
    Extract curtailment windows from SCADA per turbine.

    Args:
        SCADA_setpoint: DataFrame containing the setpoint signal

        difference_ratio: ratio to determine maximum deviation after which
        a setpoint change is considered new curtailment

        additional_flags: if True, additional information such as high wind
        ride through will be added, thus save iterations

        count_zero_as_curtailment: if True, setpoint values of 0 and 1 will be
        counted as curtailment

        fill_missing_data: if True, missing setpoint values are filled with 0 for
        all timestamps present for at least one turbine. Otherwise remaining
        timestamps are treated as non-existing in single-turbine operations.

        SCADA_Wind: ambient wind scada data, only needed if additional_flags is
        True

        curtailment_reports: Park curtailment reports, only necessary if
        additional_flags is set

        turbine_events: turbine event data for all turbines, only necessary if
        additional_flags is set

        missing_as_rated: if True, missing setpoint values will be treated as
        operations, i.e. they will be set to rated power

    Returns:
        list of dataframes with curtailment events per turbine and list of
        dataframes of flagged timeseries per turbine
    """

    setpoint = SCADA_setpoint
    max_val = setpoint['Value'].max()

    # If missing data has to be filled
    if fill_missing_data:

        # Default: fill with zeros
        if missing_as_rated:
            setpoint = setpoint.pivot_table(index=['Timestamp'], columns=['Turbine'], fill_value=max_val)

        # Optionally: fill with rated power: business as usual
        else:
            setpoint = setpoint.pivot_table(index=['Timestamp'], columns=['Turbine'], fill_value=0)

    # Otherwise: don't fill missing data
    else:
        setpoint = setpoint.pivot_table(index=['Timestamp'], columns=['Turbine'])
    setpoint = setpoint.stack().swaplevel(i='Turbine', j='Timestamp')

    names = SCADA_setpoint["Turbine"].unique()
    names = np.sort(names)
    curtailment_widows = []
    turbine_with_curtailments = []

    for name in tqdm(names):


        turbine_data = setpoint.xs(name, level='Turbine').copy()

        previous_val = turbine_data['Value'].shift()
        current_val = turbine_data['Value']

        # isChanged is true when the current value is outside a (user-specified) range
        # around the previous value. It is also true if a value close to 0|1
        # goes to 0|1 and if a low value comes right after a 0|1
        # (unless count_zero_as_curtailment is set)
        turbine_data['isChanged'] = \
            ~within_margin(previous_val, current_val, max_val, difference_ratio=difference_ratio) | \
            ((current_val == max_val) & (previous_val != max_val)) | \
            ((current_val != max_val) & (previous_val == max_val)) | \
            ((((current_val - 2 < 0) & (previous_val - 2 >= 0)) | \
              ((previous_val - 2 < 0) & (current_val - 2 >= 0))) & ~count_zero_as_curtailment)

        # isCurtailed is True whenever a turbine's setpoint is not at rated power and
        # not at 0 or 1 (unless count_zero_as_curtailment is set)
        turbine_data['isCurtailed'] = (turbine_data['Value'] != max_val) & \
                                      (((turbine_data['Value'] != 0) & (turbine_data['Value'] != 1)) \
                                       | count_zero_as_curtailment)

        # StartCurtailment is whenever the setpoint changes and
        # the turbine is curtailed
        turbine_data['StartCurtailment'] = turbine_data['isCurtailed'] & \
                                           turbine_data['isChanged']

        # A curtailment window ends when the next value is different and the current
        # value is not curtailed. If the data ends with curtailment, an End is added
        # too
        turbine_data['EndCurtailment'] = \
            turbine_data['isCurtailed'] & turbine_data['isChanged'].shift(-1) | \
            turbine_data['isCurtailed'] & turbine_data['isChanged'].shift(-1).isna()

        turbine_curtailments = pd.DataFrame()
        # Get first timestamp of active curtailment event
        turbine_curtailments['Start'] = turbine_data[turbine_data['StartCurtailment'] == True].index

        # Get last timestamp of active curtailment event
        turbine_curtailments['End'] = turbine_data[turbine_data['EndCurtailment'] == True].index

        turbine_curtailments = turbine_curtailments.set_index(['Start'])

        if additional_flags:
            # Add HWRT to turbine data
            turbine_data = high_wind_ride_through(turbine_data, \
                                                  SCADA_wind[SCADA_wind['Turbine'] == name])

            # Add HWRT to curtailment windows, if the start of the curtailment has
            # high wind (possibly problematic)
            turbine_curtailments['HWRT'] = \
                turbine_data.loc[turbine_curtailments.index]['HWRT'].values

            # Add technician curtailment flags to turbine data and curtailment data
            turbine_data = technician_curtailment(turbine_data)

            # A curtailment window is flagged as technician curtailment if the first
            # instance of curtailment within its respective turbine's time series data
            # is flagged as technician curtailment (possibly problematic)
            turbine_curtailments['isTechnicianCurtailment'] = \
                turbine_data.loc[turbine_curtailments.index]['isTechnicianCurtailment']

            # Add a flag for reported curtailment
            efficient_curtailment_overlap_calculation(turbine_data, curtailment_reports, timeseries=True)
            efficient_curtailment_overlap_calculation(turbine_curtailments, curtailment_reports, timeseries=False)

            # Expand columns (preparation for next step)
            turbine_curtailments = expand_dummy_event_columns(turbine_curtailments, turbine_events)

            # Expand turbine curtailment to include event information
            single_turbine_events = turbine_events[turbine_events.Turbine == name].sort_values(by=['From'])
            single_turbine_events = single_turbine_events.set_index('From')
            efficient_event_overlap_calculation(turbine_curtailments, single_turbine_events)

        # Add turbine curtailments to the list of curtailment windows per turbine
        # The turbines in the list are sorted lexicographically, so GDT001 is the
        # first
        curtailment_widows.append(turbine_curtailments)
        turbine_with_curtailments.append(turbine_data)

        # update progress bar
    return curtailment_widows, turbine_with_curtailments

def print_validation_statistics_for_curtailment(mydataset: pd.DataFrame, validator: pd.DataFrame) :
  """
  Evaluates precision and recall comparing existing analytics curtailment model
  outcomes to this model.

  Args:
      mydataset: DataFrame containing flagged timeseries
      validator: DataFrame used for validating results

  Returns:
      pretty-printable DataFrames
  """

  # Trim dates
  validator = validator[validator.index >= mydataset.index.min()].copy()
  validator = validator[validator.index <= mydataset.index.max()]

  # Add boolean columns on the validator dataset to facilitate FP and FN calculation
  validator['isCurtailed'] = (validator.label == 'reported curtailment') | (validator.label == 'unreported curtailment')

  validator['isReported'] = (validator.label == 'reported curtailment')

  validator = validator[['isCurtailed', 'isReported']]

  # Print length statistics
  print("length of validation dataset: ", len(validator), " start date: ", validator.index.min(), " end date: ", validator.index.max())
  print("length of dataset to validate: ", len(mydataset), " start date: ", mydataset.index.min(), " end date: ", mydataset.index.max())
  mergedindex = validator.index.intersection(mydataset.index)
  print("length of intersected index: ", len(mergedindex), " start date: ", mergedindex.min(), " end date: ", mergedindex.max())

  validator = validator[validator.index.isin(mergedindex)]
  mydataset = mydataset[mydataset.index.isin(mergedindex)]

  TP_overall = validator[(validator.isCurtailed == True) & (mydataset.isCurtailed == True)]
  TN_overall = mydataset[(validator.isCurtailed == False) & (mydataset.isCurtailed == False)]
  FP_overall = mydataset[(validator.isCurtailed == True) & (mydataset.isCurtailed == False)]
  FN_overall = mydataset[(validator.isCurtailed == False) & (mydataset.isCurtailed == True)]
  total_overall = len(TP_overall)+len(TN_overall)+len(FP_overall)+len(FN_overall)
  TPR_overall = len(TP_overall)/(len(TP_overall)+len(FN_overall))
  TNR_overall = len(TN_overall)/(len(TN_overall)+len(FP_overall))
  accuracy_overall = (len(TP_overall)+len(TN_overall))/(total_overall)
  balanced_accuracy_overall = (TPR_overall+TNR_overall)/2
  precision_overall = len(TP_overall)/(len(TP_overall)+len(FP_overall))

  print("\ngeneral curtailment statistics: ")
  print("TP: ", len(TP_overall ))
  print("TN: ", len(TN_overall))
  print("FP: ", len(FP_overall))
  print("FN: ", len(FN_overall))
  print("precision: ", precision_overall,
        " recall/sensitivity/true positive rate: ", TPR_overall,
        " specificity/true negative rate ", TNR_overall)
  print("accuracy: ", accuracy_overall, " balanced accuracy: ", balanced_accuracy_overall)

  TP_reported = mydataset[(validator.isReported == True) &
                          (validator.isCurtailed == True) &
                          (mydataset.Reported == True) &
                          (mydataset.isCurtailed == True)]
  TN_reported = mydataset[(validator.isReported == False) &
                          (validator.isCurtailed == False) &
                          (mydataset.Reported == False) &
                          (mydataset.isCurtailed == False)]
  FP_reported = mydataset[(validator.isReported == False) &
                          (validator.isCurtailed == False) &
                          (mydataset.Reported == True) &
                          (mydataset.isCurtailed == True)]
  FN_reported = mydataset[(validator.isReported == True) &
                          (validator.isCurtailed == True) &
                          (mydataset.Reported == False) &
                          (mydataset.isCurtailed == False)]
  total_reported = len(TP_reported)+len(TN_reported)+len(FP_reported)+len(FN_reported)
  TPR_reported = len(TP_reported)/(len(TP_reported)+len(FN_reported))
  TNR_reported = len(TN_reported)/(len(TN_reported)+len(FP_reported))
  accuracy_reported = (len(TP_reported)+len(TN_reported))/(total_reported)
  balanced_accuracy_reported = (TPR_reported+TNR_reported)/2
  precision_reported = len(TP_reported)/(len(TP_reported)+len(FP_reported))

  print("\nreported curtailment statistics: ")
  print("TP: ", len(TP_reported ))
  print("TN: ", len(TN_reported))
  print("FP: ", len(FP_reported))
  print("FN: ", len(FN_reported))
  print("precision: ", precision_reported,
        " recall/sensitivity/true positive rate: ", TPR_reported,
        " specificity/true negative rate ", TNR_reported)
  print("accuracy: ", accuracy_reported, " balanced accuracy: ", balanced_accuracy_reported)

  TP_unreported = mydataset[(validator.isCurtailed == True) &
                            (validator.isReported == False) &
                            (mydataset.isCurtailed== True) &
                            (mydataset.Reported == False)]
  TN_unreported = mydataset[(validator.isCurtailed == False) &
                            (validator.isReported == False) &
                            (mydataset.isCurtailed== False) &
                            (mydataset.Reported == False)]
  FP_unreported = mydataset[(validator.isCurtailed == False) &
                            (validator.isReported == False) &
                            (mydataset.isCurtailed== True) &
                            (mydataset.Reported == False)]
  FN_unreported = mydataset[(validator.isCurtailed == True) &
                            (validator.isReported == False) &
                            (mydataset.isCurtailed== False) &
                            (mydataset.Reported == False)]
  total_unreported = len(TP_unreported)+len(TN_unreported)+len(FP_unreported)+len(FN_unreported)
  try:
    TPR_unreported = len(TP_unreported)/(len(TP_unreported)+len(FN_unreported))
  except:
    TPR_unreported = np.nan
  try:
    TNR_unreported = len(TN_unreported)/(len(TN_unreported)+len(FP_unreported))
  except:
    TNR_unreported = np.nan
  accuracy_unreported = (len(TP_unreported)+len(TN_unreported))/(total_unreported)
  balanced_accuracy_unreported = (TPR_unreported+TNR_unreported)/2
  precision_unreported = len(TP_unreported)/(len(TP_unreported)+len(FP_unreported))

  print("\nunreported curtailment statistics: ")
  print("TP: ", len(TP_unreported ))
  print("TN: ", len(TN_unreported))
  print("FP: ", len(FP_unreported))
  print("FN: ", len(FN_unreported))
  print("precision: ", precision_unreported,
        " recall/sensitivity/true positive rate: ", TPR_unreported,
        " specificity/true negative rate ", TNR_unreported)
  print("accuracy: ", accuracy_unreported, " balanced accuracy: ", balanced_accuracy_unreported)

  output_overall = pd.DataFrame()
  output_overall['TP'] = [len(TP_overall)]
  output_overall['TN'] = len(TN_overall)
  output_overall['FP'] = len(FP_overall)
  output_overall['FN'] = len(FN_overall)
  output_overall['TPR'] = TPR_overall
  output_overall['TNR'] = TNR_overall
  output_overall['precision'] = precision_overall
  output_overall['accuracy'] = accuracy_overall
  output_overall['balanced accuracy'] = balanced_accuracy_overall

  output_reported = pd.DataFrame()
  output_reported['TP'] = [len(TP_reported)]
  output_reported['TN'] = len(TN_reported)
  output_reported['FP'] = len(FP_reported)
  output_reported['FN'] = len(FN_reported)
  output_reported['TPR'] = TPR_reported
  output_reported['TNR'] = TNR_reported
  output_reported['precision'] = precision_reported
  output_reported['accuracy'] = accuracy_reported
  output_reported['balanced accuracy'] = balanced_accuracy_reported

  output_unreported = pd.DataFrame()
  output_unreported['TP'] = [len(TP_unreported)]
  output_unreported['TN'] = len(TN_unreported)
  output_unreported['FP'] = len(FP_unreported)
  output_unreported['FN'] = len(FN_unreported)
  output_unreported['TPR'] = TPR_unreported
  output_unreported['TNR'] = TNR_unreported
  output_unreported['precision'] = precision_unreported
  output_unreported['accuracy'] = accuracy_unreported
  output_unreported['balanced accuracy'] = balanced_accuracy_unreported


  return output_overall, output_reported, output_unreported


def compute_duration_coverage(turbine_curtailments: pd.DataFrame, turbine_events: pd.DataFrame,
                              filter_known_curtailment=False, filter_technician=True,
                              minimum_duration=None):
    """
    Rank events according to how much of their total duration is covered
    by curtailment.

    Args:
        turbine_curtailments: DataFrame containing flagged curtailment data for
        one turbine. All event flags required

        turbine_events: raw event DataFrame

        filter_known_curtailment: if True, only curtailment with unknown causes
        will be used. NB: unknown curtailment triggers might still happen while
        a reported curtailment is active or while wind is high.

        compute_difference_5112: if True difference between total duration of 5112
        events and durations covered by curtailment will be computed. This is
        mainly used to check which turbines are more affected by 5112 (for dev
        purposes)

        filter_technician = include technician curtailment in the filtering process

        minimum_duration = threshold for dropping potentially less relevant
        low-duration events.

    Returns:
        DataFrame with rankeable events (and a debug dataframe containing
        information pertaining event 5112)
    """

    codes = turbine_events['ErrorCode'].unique()
    codes = np.sort(codes)

    discovered_curtailments = turbine_curtailments

    covered_ratio_data = pd.DataFrame(data=0, index=codes, columns=['total_covered_duration'])
    col_names = [str(code) + '_Coverage_Duration' for code in codes]


    # Iterate over list of turbine curtailments
    for curtailment_data in tqdm(discovered_curtailments):

        # Filter data if causes already known and if parameter set
        if filter_known_curtailment:
            if filter_technician:
                tmp = curtailment_data[~curtailment_data.Reported &
                                       ~curtailment_data.HWRT &
                                       ~curtailment_data.isTechnicianCurtailment].agg(
                    ['sum'])[col_names]
            else:
                tmp = curtailment_data[~curtailment_data.Reported &
                                       ~curtailment_data.HWRT].agg(
                    ['sum'])[col_names]
        else:
            tmp = curtailment_data.agg(['sum'])[col_names]

        tmp.columns = codes
        tmp = tmp.transpose()
        tmp.columns = ['total_covered_duration']
        covered_ratio_data = covered_ratio_data + tmp


    covered_ratio_data['total_duration'] = turbine_events.groupby(by=['ErrorCode']).Duration.sum().dt.total_seconds()

    covered_ratio_data['total_coverage_ratio'] = (
                covered_ratio_data['total_covered_duration'] / covered_ratio_data['total_duration'])
    covered_ratio_data['total_coverage_ratio'].fillna(0)

    if minimum_duration != None:
        covered_ratio_data = covered_ratio_data[covered_ratio_data.total_duration > minimum_duration]

    return covered_ratio_data


def daily_density_events_by_mutual_information(turbine_curtailments: pd.DataFrame, turbine_events: pd.DataFrame, repeats=10):
    """
    Compute mutual information scores for events based on daily density

    Args:
        turbine_curtailments: DataFrame containing flagged curtailment data for
        one turbine. All event flags required

        turbine_events: raw event DataFrame

        repeats: how often to perform mutual information estimation (not yet ported from old version)

    Returns:
        list of mutual information per event.
    """

    turbine_curtailments_list = turbine_curtailments.copy()

    codes = turbine_events['ErrorCode'].unique()
    codes = np.sort(codes)

    names = turbine_events['Turbine'].unique()
    names = np.sort(names)

    start = turbine_curtailments_list[0].index.min().floor(freq='24H')
    end = turbine_curtailments_list[0].index.max().floor(freq='24H')
    nrdays = (end - start).days

    repeater = np.zeros((len(codes), len(turbine_curtailments_list), repeats))

    indexnames = [start + timedelta(days=day) for day in range(nrdays)]

    covered_ratio_data = pd.DataFrame(data=0, index=indexnames, columns=['End'])
    covered_ratio_data['End'] = covered_ratio_data.index + timedelta(minutes=(23 * 60 + 59))
    covered_ratio_data = [covered_ratio_data for i in range(80)]
    covered_ratio_data = expand_dummy_event_columns(covered_ratio_data, turbine_events)

    i = 0
    for turbine_curtailments in turbine_curtailments_list:
        single_turbine_events = turbine_events[turbine_events['Turbine'] == names[i]].sort_values(by=['From']).copy()
        single_turbine_events = single_turbine_events.set_index('From')
        current = covered_ratio_data[i].copy()
        efficient_event_overlap_calculation(current, single_turbine_events)

        col_names = [str(code) + '_Coverage_Duration' for code in codes]

        current = current[col_names]

        turbine_curtailments = turbine_curtailments[~turbine_curtailments.Reported &
                                                    ~turbine_curtailments.HWRT &
                                                    ~turbine_curtailments.isTechnicianCurtailment].copy()
        turbine_curtailments['To'] = turbine_curtailments['End']
        turbine_curtailments['ErrorCode'] = 'Y'
        turbine_curtailments = turbine_curtailments[['To', 'ErrorCode']]

        currentcurt = pd.DataFrame(data=0, index=indexnames, columns=['End'])
        currentcurt['End'] = currentcurt.index + timedelta(minutes=(23 * 60 + 59))
        currentcurt['Y_Curtailment_Coverage'] = 0
        currentcurt['Y_Coverage_Duration'] = 0
        currentcurt['Y_Coverage_Ratio'] = 0
        efficient_event_overlap_calculation(currentcurt, turbine_curtailments)

        for repeat in range(repeats):
            repeater[:, i, repeat] = mutual_info_classif(current, currentcurt.Y_Coverage_Duration)

        i += 1

    avg_turbine_result = repeater.mean(axis=2)
    result = avg_turbine_result.mean(axis=1)

    ranked_list = pd.DataFrame(data=0, index=codes, columns=['mutual information scores'])
    ranked_list['mutual information scores'] = result
    ranked_list['code'] = ranked_list.index
    ranked_list = ranked_list.sort_values(by=['mutual information scores'], ascending=False)
    ranked_list['rank'] = range(0, len(ranked_list))
    ranked_list = ranked_list.set_index('rank')
    return ranked_list

def produce_event_ranking(covered_ratio_data: pd.DataFrame, turbine_events: pd.DataFrame):
    """
    Takes the output of the duration coverage function and make the dataframe pretty

    Args:
      covered_ratio_data: DataFrame containing the output of the duration coverage function
      turbine_events: turbine event dataset

    Returns:
      pretty-printable dataframe
    """

    ranking_zero = covered_ratio_data.sort_values(by=['total_coverage_ratio'], ascending=False)
    ranking_zero['rank'] = range(1, len(ranking_zero)+1)

    topdesc = [turbine_events[turbine_events.ErrorCode ==value].iloc[0].Text for value in ranking_zero.index]
    ranking_zero['description'] = topdesc
    ranking_zero['code'] = ranking_zero.index

    return ranking_zero[['rank', 'code', 'total_coverage_ratio', 'description']]