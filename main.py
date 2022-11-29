import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats
from distfit import distfit
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter


class ScheduleInvestigator:
    label_start = 'time:timestamp'
    label_end = 'end:timestamp'
    label_caseid = 'case:concept:name'
    label_resources = 'org:resource'
    label_jobs = 'concept:name'
    label_duration = 'duration'
    label_events = 'EventID'
    schedule_is_feasible = None
    feasibility_infringements = 'Not checked'
    schedule_is_non_delay = None
    schedule_is_active = None
    r_a_earth_movers = None

    def __init__(self, schedule, feasibility_measure='simple'):
        self.feasibility_measure = feasibility_measure
        self.schedule = schedule

    def investigate(self):
        # place to add
        self.is_feasible()

    def is_feasible(self):
        log = self.schedule[[self.label_start, self.label_end, self.label_events, self.label_resources]].copy()
        grouped_log = log.groupby(self.label_resources)
        infeasible_positions = 0
        dict_of_infringed_events = dict()  # a dict of observed event id(key): list of infringed events (value)
        for resource, df_resource in grouped_log:
            if df_resource.empty:
                continue
            records = df_resource.to_dict(orient='index')
            for row in records:
                values = records[row]
                id = values[self.label_events]
                start = values[self.label_start]
                end = values[self.label_end]
                if self.feasibility_measure == 'multiple':
                    df_inf = df_resource[(df_resource[self.label_start] < end) &
                                         (df_resource[self.label_start] > start)]
                elif self.feasibility_measure == 'simple':
                    df_inf = df_resource[(df_resource[self.label_start] < end) &
                                         (df_resource[self.label_start] > start)]
                    if not df_inf.empty:
                        infeasible_positions = infeasible_positions + 1
                        print_report = False
                        if print_report:
                            print('resource involved in intersection ', resource)
                            print('start', values[self.label_start], 'end: ', values[self.label_end])
                            print(df_inf.iloc[0][self.label_start], df_inf.iloc[0][self.label_end])
                        continue
                if df_inf.empty:
                    continue
                infringed_events = df_inf[self.label_events].to_list()
                dict_of_infringed_events[id] = infringed_events
                infeasible_positions = infeasible_positions + len(infringed_events)

        if infeasible_positions > 0:
            self.schedule_is_feasible = False
        else:
            self.schedule_is_feasible = True
        self.feasibility_infringements = infeasible_positions


class Planning:
    base_log = pd.DataFrame([])
    base_log_updated_ends = pd.DataFrame([])
    base_log_updated_and_revised = pd.DataFrame([])
    schedule_updated_ends = pd.DataFrame([])

    lower_limit_for_events_per_cell = 5
    distributions_to_check_for = list(['norm', 'expon', 'gamma', 'beta'])
    method_to_estimate = None
    num_of_std_s = 1

    r_a_matrix_raw = pd.DataFrame([])
    r_a_matrix_fitted_objects = pd.DataFrame([])
    r_a_matrix_values = pd.DataFrame([])

    label_start = 'time:timestamp'
    label_end = 'end:timestamp'
    label_duration = 'duration'
    label_case = 'case:concept:name'
    label_resources = 'org:resource'
    label_jobs = 'concept:name'

    def __init__(self, historic_data, schedule, safety_percent=0.8, method='dist fitting', n_of_std_s=1):
        self.base_log = historic_data
        self.schedule = schedule
        self.safety_percent = safety_percent
        self.method_to_estimate = method
        self.num_of_std_s = n_of_std_s
        self.report_on_fitted_distributions = None

    def start_pipeline(self, in_measure='mean', n_of_std_s=1):
        self.base_log[self.label_duration] = self.base_log[self.label_end] - self.base_log[self.label_start]
        self.extract_r_a_values()
        if self.method_to_estimate == 'dist fitting':
            self.fit_distributions()
            self.investigate_fitted_distributions()
            self.compute_percentiles()
        elif self.method_to_estimate == 'standard deviations':
            self.num_of_std_s = n_of_std_s
            self.calculate_std_s(measure=in_measure)
        self.revise_end_timestamps(use_estimated_values=True)

    def vary_percentiles(self, percentile, revise_ends=True):
        self.safety_percent = percentile
        self.compute_percentiles()
        if revise_ends:
            self.revise_end_timestamps(use_estimated_values=True)

    def extract_r_a_values(self):
        r_a_groups = self.base_log.copy().groupby([self.label_jobs, self.label_resources])
        cols = list(self.base_log.copy()[self.label_jobs].unique())
        idxs = list(self.base_log.copy()[self.label_resources].unique())
        self.r_a_matrix_raw = pd.DataFrame([], columns=cols, index=idxs)
        for name, r_a_group in r_a_groups:
            self.r_a_matrix_raw.loc[name[1], name[0]] = r_a_group[self.label_duration].to_list()

    def calculate_std_s(self, measure='mean'):
        self.r_a_matrix_values = pd.DataFrame(columns=self.r_a_matrix_raw.columns,
                                              index=self.r_a_matrix_raw.index)
        self.lower_limit_for_events_per_cell = 5
        for activity in self.r_a_matrix_raw.columns:
            for resource in self.r_a_matrix_raw.index:
                if not pd.isna(self.r_a_matrix_raw).loc[resource, activity]:
                    if len(self.r_a_matrix_raw.loc[resource, activity]) == 0:
                        print('i am here')
                        continue
                    elif len(self.r_a_matrix_raw.loc[resource, activity]) < self.lower_limit_for_events_per_cell:
                        self.r_a_matrix_values.loc[resource, activity] = pd.Series(
                            self.r_a_matrix_raw.loc[resource, activity]).mean()
                        continue
                    if pd.Series(self.r_a_matrix_raw.loc[resource, activity]).isna().any():
                        print('NANs in r_a ')
                    if measure == 'mean':
                        base_value = pd.Series(self.r_a_matrix_raw.loc[resource, activity]).mean()
                    elif measure == 'median':
                        base_value = pd.Series(self.r_a_matrix_raw.loc[resource, activity]).median()
                    else:
                        print('error, no base value calculation method')
                    standard_dev = pd.Series(self.r_a_matrix_raw.loc[resource, activity]).std()
                    duration_to_be = base_value + self.num_of_std_s * standard_dev
                    if duration_to_be < pd.Timedelta(seconds=0):
                        duration_to_be = pd.Timedelta(seconds=1)
                    self.r_a_matrix_values.at[resource, activity] = duration_to_be

    def fit_distributions(self):
        self.report_on_fitted_distributions = None
        self.r_a_matrix_fitted_objects = pd.DataFrame(columns=self.r_a_matrix_raw.columns,
                                                      index=self.r_a_matrix_raw.index)
        self.r_a_matrix_values = pd.DataFrame(columns=self.r_a_matrix_raw.columns,
                                              index=self.r_a_matrix_raw.index)
        for activity in self.r_a_matrix_raw.columns:
            for resource in self.r_a_matrix_raw.index:
                if not pd.isna(self.r_a_matrix_raw).loc[resource, activity]:
                    if len(self.r_a_matrix_raw.loc[resource, activity]) == 0:
                        print('i am here')
                        continue
                    elif len(self.r_a_matrix_raw.loc[resource, activity]) < self.lower_limit_for_events_per_cell:
                        self.r_a_matrix_fitted_objects.loc[resource, activity] = pd.Series(self.r_a_matrix_raw.loc[resource, activity]).median()
                        continue
                    dist = distfit(distr=self.distributions_to_check_for)
                    if pd.Series(self.r_a_matrix_raw.loc[resource, activity]).isna().any():
                        print('NANs in r_a ')
                    dist_fit_values = pd.Series(self.r_a_matrix_raw.loc[resource, activity]) / pd.Timedelta(seconds=1)
                    dist_fit_values = dist_fit_values.to_list()
                    dist.fit_transform(np.array(dist_fit_values), verbose=1)
                    self.r_a_matrix_fitted_objects.at[resource, activity] = dist.model

    def investigate_fitted_distributions(self):
        report_on_fitted_distributions = list()
        for activity in self.r_a_matrix_fitted_objects.columns:
            for resource in self.r_a_matrix_fitted_objects.index:
                try:
                    self.r_a_matrix_fitted_objects.loc[resource, activity]['name']
                except:
                    pass
                else:
                    report_on_fitted_distributions.append(self.r_a_matrix_fitted_objects.at[resource, activity]['name'])
        report_on_fitted_distributions = pd.Series(report_on_fitted_distributions).value_counts()
        self.report_on_fitted_distributions = report_on_fitted_distributions

    def compute_percentiles(self):
        if self.r_a_matrix_fitted_objects.empty:
            raise 'error'
        for activity in self.r_a_matrix_fitted_objects.columns:
            for resource in self.r_a_matrix_fitted_objects.index:
                if not pd.isna(self.r_a_matrix_fitted_objects).loc[resource, activity]:
                    if len(self.r_a_matrix_raw.loc[resource, activity]) < self.lower_limit_for_events_per_cell:
                        val_one = pd.Series(self.r_a_matrix_raw.loc[resource, activity]).mean()
                    else:
                        model = self.r_a_matrix_fitted_objects.loc[resource, activity]['model']
                        val_one = model.cdf(x=0)  # clear the values below zero
                        val_one = model.ppf(q=self.safety_percent * (1 - val_one) + val_one)
                        val_one = pd.to_timedelta(val_one, unit='s')
                        if val_one < pd.to_timedelta(0):
                            print('error 1')
                    self.r_a_matrix_values.loc[resource, activity] = val_one

    def revise_end_timestamps(self, use_estimated_values=True):
        schedule_updated_ends = pd.DataFrame([])
        r_a_groups = self.schedule.groupby([self.label_jobs, self.label_resources])
        for name, r_a_df in r_a_groups:
            if r_a_df.empty:
                continue
            if use_estimated_values:
                if name[1] not in self.r_a_matrix_values.index:
                    value = pd.Series((self.r_a_matrix_values[name[0]].dropna() / pd.Timedelta(seconds=1))).mean(numeric_only=False)
                    r_a_df[self.label_duration] = pd.to_timedelta(value, unit='s')
                else:
                    r_a_df[self.label_duration] = self.r_a_matrix_values[name[0]][name[1]]
            r_a_df.loc[:, self.label_resources] = name[1]
            r_a_df.loc[:, self.label_jobs] = name[0]
            r_a_df[self.label_end] = r_a_df[self.label_start] + pd.to_timedelta(r_a_df[self.label_duration])
            schedule_updated_ends = pd.concat([schedule_updated_ends, r_a_df])

        schedule_updated_ends[self.label_start] = pd.to_datetime(schedule_updated_ends[self.label_start], utc=True)
        schedule_updated_ends[self.label_end] = pd.to_datetime(schedule_updated_ends[self.label_end], utc=True)

        self.schedule_updated_ends = schedule_updated_ends


def vary_percentiles(sample_intern, schedule_intern, percentiles_intern=[0.4, 0.6], suffix=''):
    df_results = pd.DataFrame()

    plan_object = Planning(historic_data=sample_intern, schedule=schedule_intern,
                           safety_percent=percentiles_intern[0], method='dist fitting')
    plan_object.start_pipeline()
    for perc in percentiles_intern:
        plan_object.vary_percentiles(percentile=perc, revise_ends=True)
        plan_object.start_pipeline()
        schedule_with_revised_ends = plan_object.schedule_updated_ends

        time_start = datetime.datetime.now()
        investigator = ScheduleInvestigator(schedule=schedule_with_revised_ends, feasibility_measure='simple')
        investigator.investigate()
        time_end = datetime.datetime.now()
        comp_time = (time_end - time_start).total_seconds()
        df_results = df_results.append({'percentile': perc,
                                        f'feasible {suffix}': bool(investigator.is_feasible),
                                        f'Schedule {suffix}': investigator.feasibility_infringements,
                                        f'# tasks {suffix}': schedule_with_revised_ends.shape[0],
                                        f'computation time {suffix}': (time_end - time_start).total_seconds()
                                        },
                                       ignore_index=True
                                       )
        print('experiment name: ', suffix, 'percentile: ', perc)
        print('time: ', comp_time, 'number of infringements: ', investigator.feasibility_infringements)
    df_results.to_csv('results_percentiles.csv')

    return df_results


def merge_ends_to_start(log, label_lifecycle='lifecycle:transition', label_start='start', label_end='complete'):
    log = log[log[label_lifecycle].isin([label_start, label_end])].sort_values('time:timestamp')
    log2 = log.groupby('case:concept:name')
    start_ends = list()
    for case_id, case_log in log2:
        case_log_dict = case_log.to_dict('records')
        incomplete_events = list()
        # print(case_log)
        # print(pd.DataFrame.from_dict(data=case_log_dict))
        for event in case_log_dict:
            if event[label_lifecycle] == label_start:
                incomplete_events.append(event)
            elif event[label_lifecycle] == label_end:
                # here complete only
                # check if event has a correspondig start
                for prev_event in incomplete_events:
                    if prev_event['concept:name'] == event['concept:name']:
                        prev_event['end:timestamp'] = event['time:timestamp']
                        start_ends.append(prev_event)
                        incomplete_events.remove(prev_event)

    df_start_ends = pd.DataFrame.from_records(data=start_ends)
    df_start_ends['duration'] = (df_start_ends['end:timestamp'] - df_start_ends['time:timestamp'])
    return df_start_ends


def transform_xes_to_csv(path, name):
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    log = xes_importer.apply(path, variant=variant, parameters=parameters)
    df_log = xes_importer.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    df_log.to_csv(f'EventData/{name}.csv')
    print('successfully saved: -->', name)
    return df_log


def csv_or_xes(mode='csv'):
    if mode == 'csv':
        log_bpi17_ = pd.read_csv('EventData/BPI17.csv')
        log_bpi17_['time:timestamp'] = pd.to_datetime(log_bpi17_['time:timestamp'])
        if 'end:timestamp' in log_bpi17_.columns:
            log_bpi17_['end:timestamp'] = pd.to_datetime(log_bpi17_['end:timestamp'])
    elif mode == 'xes':
        log_bpi17_ = transform_xes_to_csv(path='EventData/BPI2017.xes', name='BPI17')
    else:
        return 'error'
    return log_bpi17_


def measure_fitting_duration(log_intern, sizes=range(1500, 5000, 500), n_experiments=5, visualize=True):
    rows_list = list()
    for size in sizes:

        log_sized = log_intern.sample(n=size, replace=True)
        for run in range(0, n_experiments):
            print(size, run)
            plan = Planning(historic_data=log_sized, schedule=log_intern.tail(50))
            plan.extract_r_a_values()
            start = pd.Timestamp.now()
            plan.fit_distributions()
            end = pd.Timestamp.now()
            raw_value_dist_fitting = end - start
            start = pd.Timestamp.now()
            plan.compute_percentiles()
            end = pd.Timestamp.now()
            raw_value_compute_percentiles = end - start
            plan_object = Planning(historic_data=log_sized, schedule=log_intern.tail(50),
                                   n_of_std_s=1, method='standard deviations')
            start = pd.Timestamp.now()
            plan_object.calculate_std_s(measure='mean')
            end = pd.Timestamp.now()
            raw_value_compute_stds = end - start
            rows_list.append(dict({'number of events': size,
                                   'raw value dist fitting': raw_value_dist_fitting.total_seconds(),
                                   'raw value percentile re-computation': raw_value_compute_percentiles.total_seconds(),
                                   'raw value compute stds': raw_value_compute_stds.total_seconds()
                                   }))
    results = pd.DataFrame(rows_list)
    print('head ', results.head(6))

    if visualize:
        df_results_aggregated = results.groupby(by='number of events')['raw value dist fitting',
                                                                       'raw value compute stds'].mean()
        df_results_aggregated = df_results_aggregated.reset_index()

        fig = px.line(df_results_aggregated, x=df_results_aggregated['number of events'], y=['raw value dist fitting',
                                                                                             'raw value compute stds'],
                      markers=True)
        fig.update_xaxes(title_text='Number of Tasks')
        fig.update_yaxes(title_text='Runtime in Seconds')
        fig.update_layout(paper_bgcolor='rgb(255, 255, 255)',
                          plot_bgcolor='rgb(255, 255, 255)',
                          width=1800, height=800,
                          legend_title='',
                          legend=dict(traceorder='normal',
                                      yanchor="top", y=0.95,
                                      xanchor="left", x=0.15),
                          font=dict(family="Times New Roman",
                                    size=28, color="Black")
                          )
        new_names = {'raw value dist fitting': 'Runtime Distribution Fitting',
                     'raw value compute stds': 'Runtime Standard Deviation Calculation'}

        fig.for_each_trace(lambda t: t.update(name=new_names[t.name],
                                              legendgroup=new_names[t.name],
                                              hovertemplate=t.hovertemplate.replace(t.name, new_names[t.name])
                                              )
                           )
        fig.show()
    return results


def experiment_percentiles(log_internal, percentiles=[0.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]):
    datasets = dict()
    intervals = [[["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-05-01T00:00:00+01:00", "2016-06-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-06-01T00:00:00+01:00", "2016-07-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-07-01T00:00:00+01:00", "2016-08-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-08-01T00:00:00+01:00", "2016-09-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-09-01T00:00:00+01:00", "2016-10-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-10-01T00:00:00+01:00", "2016-11-01T00:00:00+01:00"]]
                 ]

    df_results = pd.DataFrame()
    counter = 0
    suffix_schedule = ['05/2016', '06/2016', '07/2016', '08/2016', '09/2016', '10/2016']
    names = list()
    for interval in intervals:
        sample_start = interval[0][0]
        sample_end = interval[0][1]
        schedule_start = interval[1][0]
        schedule_end = interval[1][1]
        sample_intern = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(sample_start)) &
                                     (log_internal['time:timestamp'] < pd.to_datetime(sample_end))]
        schedule_intern = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(schedule_start)) &
                                       (log_internal['time:timestamp'] < pd.to_datetime(schedule_end))]
        print(f'schedule from {schedule_start} to {schedule_end}')
        print(f'size sample: {sample_intern.shape[0]}, size schedule: {schedule_intern.shape[0]}')
        datasets[suffix_schedule[counter]] = [sample_intern, schedule_intern]
        number_tasks = schedule_intern.shape[0]
        df_1 = vary_percentiles(sample_intern=sample_intern,
                                schedule_intern=schedule_intern,
                                percentiles_intern=percentiles,
                                suffix=suffix_schedule[counter]
                                )
        df_results['percentile'] = df_1['percentile']
        df_results[f'Schedule {suffix_schedule[counter]}'] = df_1[f'Schedule {suffix_schedule[counter]}'] / number_tasks * 100
        names.append(f'Schedule {suffix_schedule[counter]}')
        counter = counter + 1

    fig = px.line(df_results, x=df_results['percentile']*100, y=names, markers=True)
    fig.update_xaxes(title_text='Percentile to compute the End of a Task')
    fig.update_yaxes(title_text='Share of Tasks with Infeasibilities in %')
    fig.update_layout(paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      width=1800, height=800,
                      yaxis_range=[0, 100],
                      legend_title='',
                      legend=dict(traceorder='normal',
                                  yanchor="top", y=0.95,
                                  xanchor="left", x=0.15),
                      font=dict(family="Times New Roman",
                                size=28, color="Black")
                      )
    fig.show()
    return


def experiment_standard_deviations(log_internal, standard_deviations=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]):
    datasets = dict()
    intervals = [[["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-05-01T00:00:00+01:00", "2016-06-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-06-01T00:00:00+01:00", "2016-07-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-07-01T00:00:00+01:00", "2016-08-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-08-01T00:00:00+01:00", "2016-09-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-09-01T00:00:00+01:00", "2016-10-01T00:00:00+01:00"]],
                 [["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                  ["2016-10-01T00:00:00+01:00", "2016-11-01T00:00:00+01:00"]]
                 ]
    df_results = pd.DataFrame()
    counter = 0
    suffix_schedule = ['05/2016', '06/2016', '07/2016', '08/2016', '09/2016', '10/2016']
    names = list()
    df_results_1 = pd.DataFrame()
    for interval in intervals:
        sample_start = interval[0][0]
        sample_end = interval[0][1]
        schedule_start = interval[1][0]
        schedule_end = interval[1][1]
        sample_intern = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(sample_start)) &
                                     (log_internal['time:timestamp'] < pd.to_datetime(sample_end))]
        schedule_intern = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(schedule_start)) &
                                       (log_internal['time:timestamp'] < pd.to_datetime(schedule_end))]
        datasets[suffix_schedule[counter]] = [sample_intern, schedule_intern]
        number_tasks = schedule_intern.shape[0]
        suffix = suffix_schedule[counter]
        plan_object = Planning(historic_data=sample_intern, schedule=schedule_intern,
                               safety_percent=standard_deviations[0], method='standard deviations')
        df_results = pd.DataFrame()
        for std_ in standard_deviations:
            plan_object.start_pipeline(in_measure='mean', n_of_std_s=std_)
            schedule_with_revised_ends = plan_object.schedule_updated_ends
            start = datetime.datetime.now()
            investigator = ScheduleInvestigator(schedule=schedule_with_revised_ends, feasibility_measure='simple')
            investigator.investigate()
            end = datetime.datetime.now()
            comp_time = (end - start).total_seconds()
            df_results = df_results.append({'Standard Deviations': std_,
                                            f'feasible {suffix}': bool(investigator.is_feasible),
                                            f'Schedule {suffix}': investigator.feasibility_infringements / number_tasks * 100,
                                            f'# tasks {suffix}': schedule_with_revised_ends.shape[0],
                                            f'computation time {suffix}': comp_time
                                            },
                                           ignore_index=True
                                           )
            print('experiment name: ', suffix, 'Standard Deviations: ', std_)
            print('time: ', comp_time, 'number of infringements: ', investigator.feasibility_infringements, "/",
                  investigator.feasibility_infringements / number_tasks * 100, '%')
        df_results_1['Standard Deviations'] = df_results['Standard Deviations']
        df_results_1[f'Schedule {suffix_schedule[counter]}'] = df_results[f'Schedule {suffix_schedule[counter]}']
        names.append(f'Schedule {suffix_schedule[counter]}')
        counter = counter + 1
    df_results_1 = df_results_1.sort_values(by=['Standard Deviations'])
    fig = px.line(df_results_1, x='Standard Deviations', y=names, markers=True)
    fig.update_xaxes(title_text='Standard Deviations added to compute the End of a Task')
    fig.update_yaxes(title_text='Share of Tasks with Infeasibilities in %')
    fig.update_layout(paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      width=1800, height=800,
                      legend_title='',
                      yaxis_range=[0, 100],
                      legend=dict(traceorder='normal',
                                  yanchor="top", y=0.95,
                                  xanchor="left", x=0.15),
                      font=dict(family="Times New Roman",
                                size=28, color="Black")
                      )
    fig.show()
    return


def experiment_distribution_types(log_intern):
    sample_intern = log_intern[(log_intern['time:timestamp'] > pd.to_datetime("2016-01-01T00:00:00+01:00")) &
                               (log_intern['time:timestamp'] < pd.to_datetime("2016-05-01T00:00:00+01:00"))]
    plan_test = Planning(historic_data=sample_intern, schedule=log_intern)
    plan_test.start_pipeline()
    dist_fit_results = plan_test.report_on_fitted_distributions
    print(dist_fit_results)
    pie_data = pd.DataFrame()
    pie_data['Occurences'] = dist_fit_results
    newnames = {'beta': 'Beta', 'gamma': 'Gamma', 'expon': 'Exponential', 'norm': 'Normal'}
    new_index = list()

    for i in range(0, len(pie_data.index)):
        if pie_data.index[i] in newnames:
            new_index.append(newnames[pie_data.index[i]])

    pie_data.index = new_index
    fig = px.pie(pie_data, values='Occurences', names=pie_data.index)
    fig.update_layout(paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      legend_title='',
                      legend=dict(traceorder='normal'),
                      font=dict(family="Times New Roman",
                                size=28, color="Black")
                      )
    fig.show()
    return


def check_infringements(sample_intern, schedule_intern, number_of_stand_dev):
    plan_object = Planning(historic_data=sample_intern, schedule=schedule_intern,
                           n_of_std_s=number_of_stand_dev, method='standard deviations')
    plan_object.start_pipeline()
    investigator = ScheduleInvestigator(schedule=plan_object.schedule_updated_ends)
    investigator.investigate()
    print(investigator.feasibility_infringements)
    return


# log_BPI17 = csv_or_xes(mode='xes')
log_BPI17 = csv_or_xes(mode='csv')

log_BPI17 = merge_ends_to_start(log_BPI17)
print('#tasks', log_BPI17.shape[0])

# measure_fitting_duration(log_intern=log_BPI17, sizes=range(1000, 8000, 500), n_experiments=5)
measure_fitting_duration(log_intern=log_BPI17, sizes=range(1000, 2000, 500), n_experiments=1)

experiment_distribution_types(log_BPI17)

# experiment_percentiles(log_BPI17)
experiment_percentiles(log_BPI17, percentiles=[0.5, 0.6])

sample = log_BPI17[(log_BPI17['time:timestamp'] > pd.to_datetime("2016-01-01T00:00:00+01:00")) &
                   (log_BPI17['time:timestamp'] < pd.to_datetime("2016-05-01T00:00:00+01:00"))]
schedule = log_BPI17[(log_BPI17['time:timestamp'] > pd.to_datetime("2016-05-01T00:00:00+01:00")) &
                     (log_BPI17['time:timestamp'] < pd.to_datetime("2016-06-01T00:00:00+01:00"))]

check_infringements(sample_intern=sample, schedule_intern=schedule, number_of_stand_dev=-2)

# experiment_standard_deviations(log_BPI17)
experiment_standard_deviations(log_BPI17, standard_deviations=[0, 1])
