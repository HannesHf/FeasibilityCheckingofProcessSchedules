import datetime
import pandas as pd
import numpy as np
import sys
import plotly.express as px
import scipy.stats
import string
from distfit import distfit
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.algo.filtering.pandas.cases import case_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter


def scheduler(jobs, resources, r_a_matrix, start_time):
    schedule_ = pd.DataFrame(data=None)
    finish_times_resources = pd.Series([], dtype=float)
    for resource in resources:
        finish_times_resources[resource] = start_time
    ev_id = 1
    for case_id in jobs:
        trace = jobs[case_id]
        for task in trace:
            enabled_resources = r_a_matrix[task].dropna().index.to_list()
            for element in enabled_resources:
                if element not in finish_times_resources.index.values:
                    enabled_resources.remove(element)
            resource_free = finish_times_resources[[enabled_resources]]
            time = resource_free.min()
            resource_free = resource_free[resource_free == time].index[0]
            duration = r_a_matrix[task][resource_free]
            time = finish_times_resources[resource_free]

            end_time = time + duration
            dct = {'case:concept:name': case_id,
                   'eventID': f'#{ev_id}',
                   'time:timestamp': time,
                   'end:timestamp': end_time,
                   'concept:name': task,
                   'org:resource': resource_free}
            ev_id += 1
            schedule_ = schedule_.append(other=dct, ignore_index=True)
            finish_times_resources[resource_free] = end_time
    return schedule_


def illustrate_events(df0, name='', show=False, y_axis='Resource', color='Activity'):
    df = df0.copy()

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    df['end:timestamp'] = pd.to_datetime(df['end:timestamp'], utc=True)

    df['org:resource'] = df['org:resource'].astype(str)
    df = df.rename(columns={'time:timestamp': 'Start',
                            'end:timestamp': 'Finish',
                            'org:resource': 'Resource',
                            'case:concept:name': 'Case ID',
                            'concept:name': 'Activity'
                            })
    df['Task'] = df['Case ID'].astype(str) + '_' + df['Activity'].astype(str)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y=y_axis, color=color, title=name)
    fig.update_layout(legend=dict(traceorder='normal', title=color,
                                  yanchor="top", y=0.99,
                                  xanchor="right", x=0.9),
                      font=dict(family="Times New Roman",
                                size=29,
                                color="Black"),
                      paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(248, 248, 248)',
                      width=1800, height=700,
                      legend_title=''
                      )
    if show:
        fig.show()

    return


def check_resource_constraint(df_0, r_a_matrix, relaxed_check):
    # check for resource feasibility
    fulfills_constraint = True
    df_output = pd.DataFrame([], columns=df_0.columns)
    df_1 = df_0.sort_values('time:timestamp').copy()
    resources = df_1['org:resource'].unique()
    num_ress = len(resources)
    for resource in resources:
        i = 100 * np.where(resource == resources)[0] / num_ress
        sys.stdout.write("\rResource Constraints Progress: %i Percent" % i)
        sys.stdout.flush()
        df_res = df_1[df_1['org:resource'] == resource]
        if df_res.shape[0] <= 1:
            continue
        starts = df_res["time:timestamp"].to_list()[1:]  # delete first value
        ends = df_res["end:timestamp"].to_list()[:-1]  # delete last value
        infeasibilities = df_res.copy()
        infeasibilities.loc[:, 'end:timestamp_res'] = starts + [starts[-1] + pd.to_timedelta(2, unit='m')]
        infeasibilities = infeasibilities[infeasibilities["end:timestamp"] > infeasibilities['end:timestamp_res']]
        # infeasibilities can be exported or used for examination
        df_output = pd.concat([df_output, df_res])

    sys.stdout.write("\rResource Constraints Progress: FINISH     ")
    sys.stdout.flush()
    return fulfills_constraint, infeasibilities


def investigation_feasibility(df0, n=100, tag_start='time:timestamp', tag_end='end:timestamp'):
    df_result = pd.DataFrame([], columns=['#events', '#infeasibilities', 'computation_time'])
    num_events = df0.shape[0]
    results = list()
    for i in range(0, n):
        results_raw = dict()
        start = datetime.datetime.now()
        feas = is_feasible(df0)
        results_raw['#events'] = num_events
        results_raw['#infeasibilities'] = feas[1]
        results_raw['computation_time'] = (datetime.datetime.now() - start).total_seconds()
        results.append(results_raw)
    df_result = df_result.from_dict(data=results)
    return df_result


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
        # values of measure: todo binary, simple, multiple
        self.feasibility_measure = feasibility_measure
        self.schedule = schedule

    def earth_movers_distance(self, r_a_values_1, r_a_dists_objs):
        self.r_a_earth_movers = pd.DataFrame(columns=r_a_values_1.columns, index=r_a_values_1.index)

        for col in r_a_values_1.columns:
            for row in r_a_values_1.index:
                if r_a_dists_objs.notna().loc[row, col]:
                    dist = r_a_dists_objs.loc[row, col]
                    num_vals = len(r_a_values_1.loc[row, col])
                    ppfs = list()
                    for i in range(0, num_vals):
                        ppfs = ppfs + [1/(num_vals + 1) * i]
                    vals_from_dist = dist.ppf(q=ppfs)
                    sample_values = r_a_values_1.loc[row, col]
                    distance = scipy.stats.wasserstein_distance(u_values=np.array(sample_values),
                                                                v_values=np.array(vals_from_dist))
                    self.r_a_earth_movers.loc[row, col] = distance

    def investigate(self):
        # self.is_active()
        # self.is_nondelay()
        self.is_feasible()

    def is_active(self):
        # log is a DataFrame
        # no machine kept idle, while waiting for operation
        # under initial assumptions: check for empty spaces between jobs on resources, if yes, not active
        active = True
        resources = log[self.label_resources].unique()
        for resource in resources:
            sub_log = log[log[self.label_resources] == resource]
            sub_log.sort_values(by=self.label_start)
            for index in range(0, sub_log.shape[0] - 1):
                end_act1 = sub_log.iloc[index][self.label_end]
                start_act2 = sub_log.iloc[index + 1][self.label_start]
                if end_act1 > start_act2:
                    active = False
                    return active
        self.schedule_is_active = active

    def is_nondelay(self):
        # no machine kept idle, while jobs waiting for operation
        # under initial assumptions: check for empty spaces between jobs on resources, if empty spaces, is_non_delay = false
        log = self.schedule.copy()
        non_delay = True
        resources = log[self.label_resources].unique()
        for resource in resources:
            sub_log = log[log[self.label_resources] == resource]
            sub_log = sub_log.sort_values(by=[self.label_start])
            for index in range(0, sub_log.shape[0] - 1):
                end_act1 = sub_log.iloc[index][self.label_end]
                start_act2 = sub_log.iloc[index + 1][self.label_start]
                if end_act1 < start_act2:
                    non_delay = False
        self.schedule_is_non_delay = non_delay

    def is_feasible(self):
        # log is a DataFrame

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
        # self.detailled_report =
        self.feasibility_infringements = infeasible_positions


class CreateArtificialData:
    label_case = 'case:concept:name'
    label_resources = 'org:resource'
    label_jobs = 'concept:name'
    label_start = 'time:timestamp'
    label_end = 'end:timestamp'
    label_duration = 'duration'
    label_event_identifier = 'EventID'

    log_with_artificial_durations = None
    base_log = None

    r_a_matrix_raw = None
    r_a_matrix_distribution_objects = None
    # distribution fitting params
    lower_limit_for_events_per_cell = -1
    distributions_to_check_for = list(['norm', 'expon', 'gamma', 'beta'])

    def __init__(self, event_log, revise_starts=False, artificial_factors_=[0.5, 0.2]):
        self.base_log = event_log
        self.artificial_factors = artificial_factors_
        self.revise_starts = revise_starts
        cols = list(self.base_log[self.label_jobs].unique())
        idxs = list(self.base_log[self.label_resources].unique())
        self.r_a_matrix_raw = pd.DataFrame([], columns=cols, index=idxs)
        self.create_r_a_matrix_raw()
        self.revise_base_log()

    def create_r_a_matrix_raw(self):
        # fill r a matrix with random values
        # find number of values per cell -> align log afterwards as non-delay and drop cells below lower_border
        group_sizes = self.base_log.copy().groupby([self.label_jobs, self.label_resources])[self.label_start].count()

        for job, resource in group_sizes.index:
            num_of_tasks = group_sizes[job, resource]
            if num_of_tasks >= self.lower_limit_for_events_per_cell:
                # if number of cells above lower limit, use random distribution to generate new data
                if len(self.distributions_to_check_for) == 1:
                    dist = self.distributions_to_check_for[0]
                else:
                    dist = self.distributions_to_check_for[
                        np.random.randint(0, len(self.distributions_to_check_for) - 1)]
                param_1 = self.artificial_factors[0]
                param_2 = self.artificial_factors[1]
                random_values = pd.Series()

                if dist == 'norm':
                    # param_1 -> mean/expected value
                    # param_2 -> width/spread

                    while random_values.size < num_of_tasks:
                        random_values_ = pd.Series(np.random.normal(loc=param_2, scale=param_1,
                                                                    size=num_of_tasks-random_values.size)).dropna()
                        random_values = pd.concat([random_values, random_values_])
                        random_values = random_values[random_values > 0]

                elif dist == 'expon':
                    # param_1 -> mean/expected value
                    # param_2 -> beta
                    while random_values.size < num_of_tasks:
                        random_values_ = pd.Series(np.random.exponential(scale=param_1,
                                                                         size=num_of_tasks-random_values.size) * param_1).dropna()
                        random_values = pd.concat([random_values, random_values_])
                        random_values = random_values[random_values > 0]

                elif dist == 'gamma':
                    # param_1 -> location; denotes greek MY; expected value
                    # param_2 -> scale; width and spread of function
                    while random_values.size < num_of_tasks:
                        random_values_ = pd.Series(np.random.gamma(shape=param_2, scale=param_1,
                                                                   size=num_of_tasks-random_values.size)).dropna()
                        random_values = pd.concat([random_values, random_values_])
                        random_values = random_values[random_values > 0]

                elif dist == 'beta':
                    # param_1 -> alpha
                    # param_2 -> beta
                    while random_values.size < num_of_tasks:
                        random_values_ = pd.Series(np.random.beta(a=param_1, b=param_2 * 3,
                                                                  size=num_of_tasks - random_values.size)).dropna()
                        random_values = pd.concat([random_values, random_values_])
                        random_values = random_values[random_values > 0]

                else:
                    print('unknown distribution --> even distribution')
                    # param 1 lower bound
                    # param 2 upper bound
                    random_values = pd.Series()

                    random_values = np.random.random(size=num_of_tasks) * (param_2 - param_1) + param_1

                if random_values[random_values < 0].shape[0] > 0:
                    print('\n\n\n\n', dist, '\n generated negative values')
                    print(random_values)
                self.r_a_matrix_raw.loc[resource, job] = random_values.to_list()

    def revise_base_log(self):
        # depending on self.revise_starts the log might be iteratively revised (tasks begin later/earlier)
        base_log = self.base_log.copy()
        base_log[self.label_end] = np.nan

        if 'end:timestamp' not in base_log.columns:
            raise TypeError('no end in r_a col')
        if 'end:timestamp' in base_log.index:
            raise TypeError('end in index')

        if self.label_duration in base_log.columns:
            # if no durations, also no end timestamps
            has_durations = True
        else:
            base_log[self.label_duration] = np.nan
            has_durations = False

        r_a_groups = base_log.groupby([self.label_jobs, self.label_resources])
        lwad_final = pd.DataFrame([], columns=base_log.columns)

        for name, r_a_df in r_a_groups:
            if r_a_df.empty:
                continue
            if 'end:timestamp' in r_a_df.index:
                raise TypeError('end in input values')
            r_a_df.sort_values(self.label_start)
            if not has_durations:
                random_durations = self.r_a_matrix_raw[name[0]][name[1]]
                r_a_df[self.label_duration] = random_durations
                if r_a_df[self.label_duration].isnull().values.any():
                    print('nans in durations')

            if self.revise_starts:
                starts = r_a_df[self.label_start]
                ends = r_a_df[self.label_end]
                durations = r_a_df[self.label_duration]
                end = starts.iat[0]
                for ind in range(0, len(starts) - 1):
                    start = starts.iat[ind]
                    duration = durations.iat[int(ind)]
                    if end > start:
                        starts.iat[ind] = end
                        start = end
                    end = start + pd.to_timedelta(duration, unit='minutes')
                    ends.iat[ind] = end
                ends.iat[len(ends) - 1] = start + pd.to_timedelta(duration, unit='m')
                r_a_df.loc[self.label_start] = starts
                r_a_df.loc[self.label_end] = ends
            else:  # not revise starts
                if self.label_end in r_a_df.index:
                    r_a_df.drop(index=[self.label_end])
                r_a_df[self.label_end] = pd.to_timedelta(r_a_df[self.label_duration], unit='minutes') + \
                                             r_a_df[self.label_start]
                if 'end:timestamp' not in r_a_df.columns:
                    print('no end in r_a col')
            #print(r_a_df[self.label_duration].size, r_a_df[self.label_duration].isnull().values.any())
            #print(r_a_df[self.label_start].size, r_a_df[self.label_start].isnull().values.any())
            #print(r_a_df[r_a_df[self.label_start].isna()].shape[0])
            if 'end:timestamp' not in r_a_df.columns:
                raise TypeError('no end in r_a col')
            if 'end:timestamp' in r_a_df.index:
                raise TypeError('end in index')
            r_a_df.loc[:, self.label_jobs] = name[0]
            r_a_df.loc[:, self.label_resources] = name[1]
            lwad_final = pd.concat([lwad_final, r_a_df])
        #lwad_final[self.label_duration] = lwad_final.
        if 'end:timestamp' not in lwad_final.columns:
            raise TypeError('no end in r_a col')
        if 'end:timestamp' in lwad_final.index:
            raise TypeError('end in index')
        self.log_with_artificial_durations = lwad_final


class Planning:
    # versions of plans
    base_log = pd.DataFrame([])
    base_log_updated_ends = pd.DataFrame([])
    base_log_updated_and_revised = pd.DataFrame([])
    schedule_updated_ends = pd.DataFrame([])

    # distribution fitting params
    lower_limit_for_events_per_cell = 5
    distributions_to_check_for = list(['norm', 'expon', 'gamma', 'beta'])
    method_to_estimate = None
    num_of_std_s = 1

    # df versions of resource activity matrix
    r_a_matrix_raw = pd.DataFrame([])
    r_a_matrix_fitted_objects = pd.DataFrame([])
    r_a_matrix_values = pd.DataFrame([])

    # general things
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
        # extract the raw values
        self.base_log[self.label_duration] = self.base_log[self.label_end] - self.base_log[self.label_start]
        self.extract_r_a_values()
        if self.method_to_estimate == 'dist fitting':
            # perform distribution fitting and save dist objects in r_a_dist
            self.fit_distributions()
            self.investigate_fitted_distributions()
            self.compute_percentiles()
        elif self.method_to_estimate == 'standard deviations':
            self.num_of_std_s = n_of_std_s
            self.calculate_std_s(measure=in_measure)
        self.revise_end_timestamps(use_estimated_values=True)

    def vary_percentiles(self, percentile, revise_ends=True):
        self.safety_percent = percentile
        # self.base_log = self.schedule_updated_ends
        self.compute_percentiles()
        if revise_ends:
            self.revise_end_timestamps(use_estimated_values=True)

    def incoming_event(self, df_event):
        # input event as df (with one or more lines)
        self.base_log = pd.concat([self.base_log, df_event])
        # todo: trigger adapted analysis again

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
        # keeps the starting point per activity for feasibility checking
        # use_estimated values, if true, values of r_a_values matrix are used
        schedule_updated_ends = pd.DataFrame([])
        # (1) revise durations
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

        if 'end:timestamp' not in schedule_updated_ends.columns:
            print('no end columns: ')

        self.schedule_updated_ends = schedule_updated_ends

    def revise_base_log(self):
        # revise the log based on r a values and existing timestamps
        base_log_2 = pd.DataFrame(columns=self.base_log.columns)
        r_a_groups = self.base_log.groupby([self.label_jobs, self.label_resources])

        for name, r_a_df in r_a_groups:
            if r_a_df.empty:
                continue
            r_a_df.sort_values(self.label_start)

            duration_value = self.r_a_matrix_values[name[0]][name[1]]
            r_a_df.loc[:, self.label_duration] = duration_value
            r_a_df.loc[:, self.label_jobs] = name[0]
            r_a_df.loc[:, self.label_resources] = name[1]
            base_log_2 = pd.concat([r_a_df, base_log_2])

        revised_schedule = pd.DataFrame(columns=self.base_log.columns)
        r_groups = base_log_2.groupby(self.label_resources)
        for res, r_df in r_groups:
            if r_df.empty:
                continue
            starts = r_df[self.label_start]
            ends = r_df[self.label_end]
            durations = r_df[self.label_duration]
            end = starts.iat[0]

            for ind in range(0, len(starts) - 1):
                duration = durations.iat[ind]
                starts.iat[ind] = end
                start = end
                end = start + pd.to_timedelta(duration, unit='m')
                ends.iat[ind] = end
            starts.iat[len(ends) - 1] = start
            ends.iat[len(ends) - 1] = start + pd.to_timedelta(duration, unit='m')

            r_df.loc[:, self.label_start] = starts
            r_df.loc[:, self.label_end] = ends
            revised_schedule = pd.concat([revised_schedule, r_df])
        revised_schedule['time:timestamp'] = pd.to_datetime(revised_schedule['time:timestamp'], utc=True)
        revised_schedule['end:timestamp'] = pd.to_datetime(revised_schedule['end:timestamp'], utc=True)
        self.base_log_updated_and_revised = revised_schedule


def vary_percentiles(sample_intern, schedule_intern, percentiles_intern=[0.4, 0.6], suffix=''):
    df_results = pd.DataFrame()
    # historic_data is a historic process execution
    # re-estimate the durations
    plan_object = Planning(historic_data=sample_intern, schedule=schedule_intern,
                           safety_percent=percentiles_intern[0], method='dist fitting')
    plan_object.start_pipeline()
    for perc in percentiles_intern:
        # extract the log with revised timestamps
        plan_object.vary_percentiles(percentile=perc, revise_ends=True)
        plan_object.start_pipeline()
        schedule_with_revised_ends = plan_object.schedule_updated_ends
        # illustrate_events(df0=schedule_with_revised_ends, name=f'exp {exp}', show=True)

        time_start = datetime.datetime.now()
        investigator = ScheduleInvestigator(schedule=schedule_with_revised_ends, feasibility_measure='simple')
        investigator.investigate()
        time_end = datetime.datetime.now()
        comp_time = (time_end - time_start).total_seconds()
        # get results, write in table
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


def vary_emd(df_in, perc=0.4):
    df_results = pd.DataFrame()
    number_of_experiments = 100
    for exp in range(1, number_of_experiments + 1):
        df_log = df_in.copy()
        object_for_event_log = CreateArtificialData(event_log=df_log)
        df_log = object_for_event_log.log_with_artificial_durations
        obj_for_planning = Planning(historic_data=df_log, safety_percent=perc / 100)
        obj_for_planning.start_pipeline()

        # SCHEDULE REPLAY
        replay = Planning(historic_data=obj_for_planning.base_log_updated_and_revised,
                          safety_percent=perc / 100)
        replay.start_log_replay(r_a_with_dist_objects=obj_for_planning.r_a_matrix_fitted_objects)
        replay_results = ScheduleInvestigator(schedule=replay.base_log_updated_ends)
        replay_results.investigate()
        replay_feasible = replay_results.schedule_is_feasible
        replay_infringements = replay_results.feasibility_infringements
        replay_results.earth_movers_distance(r_a_values_1=obj_for_planning.r_a_matrix_raw,
                                             r_a_dists_objs=replay.r_a_matrix_fitted_objects)
        ea_matrix = replay_results.r_a_earth_movers
        s = pd.Series()
        for col in ea_matrix.columns:
            s = pd.concat([s, ea_matrix[col].dropna()], ignore_index=True)
        s_inf = s[s.isin([np.inf, -np.inf])]
        s_values = s[~s.isin([np.inf, -np.inf])]
        df_results = df_results.append({'confidence interval': perc,
                                        'feasible': bool(replay_feasible),
                                        '# infringements': replay_infringements.shape[0],
                                        '# tasks': df_log.shape[0],
                                        'mean earth movers distance': s_values.mean(),
                                        'number of emd == inf': s_inf.size
                                        },
                                       ignore_index=True
                                       )
        print(perc, exp)
    df_results.to_csv('results_emd.csv')

    fig = px.scatter(df_results.copy(), x='mean earth movers distance', y='# infringements', color='number of emd == inf',
                     symbol='number of emd == inf')
    fig.update_traces(marker_size=10)
    fig.show()


def investigate_feasibility_of_stds(data_sample, schedule,
                                    list_of_standard_deviations=[-0.5, -1, -1.5, -2, 0, 0.5, 1, 1.5, 2],
                                    n_experiments=2, mode='median', artificial_durations=False):
    df_results = pd.DataFrame()
    for num_std in list_of_standard_deviations:
        for exp in range(1, n_experiments + 1):
            if artificial_durations:
                # create new artificial data
                artificial_sample = CreateArtificialData(event_log=data_sample)
                # get data to learn from (currently = schedule)
                data_sample = artificial_sample.log_with_artificial_durations
            plan_object = Planning(historic_data=data_sample, schedule=schedule, method='standard deviations',
                                   n_of_std_s=num_std)
            plan_object.start_pipeline(in_measure=mode)
            # extract the log with revised timestamps
            schedule_with_revised_ends = plan_object.schedule_updated_ends
            # trigger feasibility check and measure time
            time_start = datetime.datetime.now()
            investigator = ScheduleInvestigator(schedule=schedule_with_revised_ends)
            investigator.investigate()
            time_end = datetime.datetime.now()
            duration = (time_end - time_start).total_seconds()
            # get results, write in table
            is_feasible = investigator.is_feasible
            infringements = investigator.feasibility_infringements
            df_results = df_results.append({'standard deviations': num_std,
                                            'feasible': bool(is_feasible),
                                            '# infringements': infringements,
                                            '# tasks': schedule.shape[0],
                                            'computation time': duration
                                            },
                                           ignore_index=True
                                           )
            print(num_std, exp)
            print(duration, infringements)

    df_results.to_csv('results/result_standard_deviations.csv')
    fig = px.scatter(df_results, x="standard deviations", y="# infringements")
    if mode == 'median':
        x_name = 'Median + x * Standard Deviations'
    elif mode == 'mean':
        x_name = 'Mean + x * Standard Deviations'
    fig.update_xaxes(title_text=x_name)
    fig.update_yaxes(title_text="Number of infeasible positions")
    fig.update_layout(title_text='Infeasibility Measures of a Schedule with estimated Durations based on Variation of Standard Deviation as Estimator',
                      paper_bgcolor='rgb(243, 243, 243)',
                      plot_bgcolor='rgb(243, 243, 243)',
                      legend_title=''
                      )
    fig.show()
    return df_results


def transform_to_format(df_in, column='lifecycle:transition'):
    # transforms an event log in shape of a dataframe with lifecycle transitions to
    # a df with start+end of event in 1 row
    df_in['time:timestamp'] = pd.to_datetime(df_in['time:timestamp'], utc=True)
    df_starts = df_in[df_in[column] == 'start'].drop(column, axis=1)
    df_ends = df_in[df_in[column] == 'complete'].drop(column, axis=1).rename(columns={'time:timestamp': 'end:timestamp'})
    df_out = pd.merge(left=df_starts, right=df_ends, on=['case:concept:name', 'org:resource', 'concept:name'])

    df_out['duration'] = df_out['end:timestamp'] - df_out['time:timestamp']
    return df_out


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
        log_rev = pd.read_csv('EventData/reviewing.csv')
        log_bpi17_ = pd.read_csv('EventData/BPI17.csv')
        log_rev['time:timestamp'] = pd.to_datetime(log_rev['time:timestamp'])
        if 'end:timestamp' in log_rev.columns:
            log_rev['end:timestamp'] = pd.to_datetime(log_rev['end:timestamp'])
        log_bpi17_['time:timestamp'] = pd.to_datetime(log_bpi17_['time:timestamp'])
        if 'end:timestamp' in log_bpi17_.columns:
            log_bpi17_['end:timestamp'] = pd.to_datetime(log_bpi17_['end:timestamp'])
    elif mode == 'xes':
        log_rev = transform_xes_to_csv(path='EventData/reviewing.xes', name='reviewing')
        log_bpi17_ = transform_xes_to_csv(path='EventData/BPI2017.xes', name='BPI17')
    else:
        return 'error'
    return [log_rev, log_bpi17_]


def measure_fitting_duration(log_intern, sizes=range(1500, 5000, 500), n_experiments=5, visualize=True):
    rows_list = list()
    for size in sizes:

        log_sized = log_intern.sample(n=size, replace=True)
        for run in range(0, n_experiments):
            print(size, run)
            plan = Planning(historic_data=log_sized, schedule=log_intern.tail(50))
            plan.extract_r_a_values()
            # count how many distributions have been fitted?
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
    results.to_csv('images/measurement_fitting_duration_values.csv')
    print('head ', results.head(6))

    if visualize:
        fig = px.scatter(results, x=results['number of events'],
                         y=['raw value dist fitting'],
                         trendline="ols")
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
                                    size=36, color="Black")
                          )
        new_names = {
            'raw value dist fitting': 'Runtime Distribution Fitting'}  # ,'raw value compute stds': 'Runtime Standard Deviation Calculation'

        fig.for_each_trace(lambda t: t.update(name=new_names[t.name],
                                              legendgroup=new_names[t.name],
                                              hovertemplate=t.hovertemplate.replace(t.name, new_names[t.name])
                                              )
                           )
        fig.show()
    return results


def experiment_percentiles(log_internal):
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
    intervals2 = [[["2016-01-01T00:00:00+01:00", "2016-05-01T00:00:00+01:00"],
                   ["2016-05-01T00:00:00+01:00", "2016-06-01T00:00:00+01:00"]],
                  [["2016-02-01T00:00:00+01:00", "2016-06-01T00:00:00+01:00"],
                   ["2016-06-01T00:00:00+01:00", "2016-07-01T00:00:00+01:00"]],
                  [["2016-03-01T00:00:00+01:00", "2016-07-01T00:00:00+01:00"],
                   ["2016-07-01T00:00:00+01:00", "2016-08-01T00:00:00+01:00"]],
                  [["2016-04-01T00:00:00+01:00", "2016-08-01T00:00:00+01:00"],
                   ["2016-08-01T00:00:00+01:00", "2016-09-01T00:00:00+01:00"]]
                  ]
    percentiles = [0.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]

    df_results = pd.DataFrame()
    counter = 0
    suffix_schedule = ['05/2016', '06/2016', '07/2016', '08/2016', '09/2016', '10/2016']
    names = list()
    for interval in intervals:
        sample_start = interval[0][0]
        sample_end = interval[0][1]
        schedule_start = interval[1][0]
        schedule_end = interval[1][1]
        sample = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(sample_start)) &
                              (log_internal['time:timestamp'] < pd.to_datetime(sample_end))]
        schedule = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(schedule_start)) &
                                (log_internal['time:timestamp'] < pd.to_datetime(schedule_end))]
        print(f'schedule from {schedule_start} to {schedule_end}')
        print(f'size sample: {sample.shape[0]}, size schedule: {schedule.shape[0]}')
        datasets[suffix_schedule[counter]] = [sample, schedule]
        number_tasks = schedule.shape[0]
        df_1 = vary_percentiles(sample_intern=sample,
                                schedule_intern=schedule,
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
                                size=36, color="Black")
                      )
    fig.show()
    return


def experiment_standard_deviations(log_internal):
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
    standard_deviations = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    counter = 0
    suffix_schedule = ['05/2016', '06/2016', '07/2016', '08/2016', '09/2016', '10/2016']
    names = list()
    df_results_1 = pd.DataFrame()
    for interval in intervals:
        sample_start = interval[0][0]
        sample_end = interval[0][1]
        schedule_start = interval[1][0]
        schedule_end = interval[1][1]
        sample = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(sample_start)) &
                              (log_internal['time:timestamp'] < pd.to_datetime(sample_end))]
        schedule = log_internal[(log_internal['time:timestamp'] > pd.to_datetime(schedule_start)) &
                                (log_internal['time:timestamp'] < pd.to_datetime(schedule_end))]
        datasets[suffix_schedule[counter]] = [sample, schedule]
        number_tasks = schedule.shape[0]

        suffix = suffix_schedule[counter]

        # historic_data is a historic process execution
        # re-estimate the durations
        plan_object = Planning(historic_data=sample, schedule=schedule,
                               safety_percent=standard_deviations[0], method='standard deviations')
        df_results = pd.DataFrame()
        for std_ in standard_deviations:
            # extract the log with revised timestamps
            plan_object.start_pipeline(in_measure='mean', n_of_std_s=std_)
            schedule_with_revised_ends = plan_object.schedule_updated_ends
            # illustrate_events(df0=schedule_with_revised_ends, name=f'exp {exp}', show=True)

            time_start = datetime.datetime.now()
            investigator = ScheduleInvestigator(schedule=schedule_with_revised_ends, feasibility_measure='simple')
            investigator.investigate()
            time_end = datetime.datetime.now()
            comp_time = (time_end - time_start).total_seconds()
            # get results, write in table
            df_results = df_results.append({'Standard Deviations': std_,
                                            f'feasible {suffix}': bool(investigator.is_feasible),
                                            f'Schedule {suffix}': investigator.feasibility_infringements / number_tasks * 100,
                                            f'# tasks {suffix}': schedule_with_revised_ends.shape[0],
                                            f'computation time {suffix}': (time_end - time_start).total_seconds()
                                            },
                                           ignore_index=True
                                           )
            print('experiment name: ', suffix, 'Standard Deviations: ', std_)
            print('time: ', comp_time, 'number of infringements: ', investigator.feasibility_infringements)
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
                                size=36, color="Black")
                      )
    fig.show()
    return


def experiment_distribution_types(log_intern):
    # check the training sample for feasibility
    sample = log_intern[(log_intern['time:timestamp'] > pd.to_datetime("2016-01-01T00:00:00+01:00")) &
                        (log_intern['time:timestamp'] < pd.to_datetime("2016-05-01T00:00:00+01:00"))]
    plan_test = Planning(historic_data=sample, schedule=log_intern)
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


data = csv_or_xes(mode='csv')
log_BPI17 = data[1]

log_BPI17 = merge_ends_to_start(log_BPI17)
print('#tasks', log_BPI17.shape[0])

# df_results1 = measure_fitting_duration(log_intern=log_BPI17, sizes=range(1000, 8000, 500), n_experiments=5)

# experiment_distribution_types(log_BPI17)

# experiment_percentiles(log_BPI17)
"""
sample = log_BPI17[(log_BPI17['time:timestamp'] > pd.to_datetime("2016-01-01T00:00:00+01:00")) &
                              (log_BPI17['time:timestamp'] < pd.to_datetime("2016-05-01T00:00:00+01:00"))]
print(sample.shape[0])
schedule = log_BPI17[(log_BPI17['time:timestamp'] > pd.to_datetime("2016-05-01T00:00:00+01:00")) &
                     (log_BPI17['time:timestamp'] < pd.to_datetime("2016-06-01T00:00:00+01:00"))]
print(schedule.shape[0])
plan_object = Planning(historic_data=sample, schedule=schedule, n_of_std_s=-2, method='standard deviations')
plan_object.start_pipeline()
investigator = ScheduleInvestigator(schedule=plan_object.schedule_updated_ends)
investigator.investigate()
print(investigator.feasibility_infringements)
"""

runtime_computation = False
if runtime_computation:
    measure_fitting_duration(log_intern=log_BPI17)

experiment_distfitting = False
if experiment_distfitting:
    experiment_standard_deviations(log_BPI17)

do_experiment_percentiles = True
if do_experiment_percentiles:
    experiment_percentiles(log_internal=log_BPI17)
