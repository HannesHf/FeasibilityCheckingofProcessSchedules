# FeasibilityCheckingofProcessSchedules

###prerequisites

upload event data in a folder    /EventData


##ScheduleInvestigator

check a schedule for infeasibilitie, i.e. overlap of tasks per one resource

1 initialize it (insert the schedule and define the feasibility measure (per default 'simple')
2 execute .investigate() (--> triggers .is_feasible() )
  .schedule_is_feasible 
      contains the boolean information
  .feasibility_infringements
      contains the number of infeasibility (depending on the measure)


##Planning

after initilizing with sample data and a schedule, it re-computes the end timestamps of the schedules based on the sample data. it uses a resource-activity matrix/dataframe to store the raw values of durations per resource and activity pair (-> r_a_matrix_raw). by applying a function on the data (e.g. distribution fitting and computation of percentiles), it computes a second resource-activity matrix (-> r_a_matrix_values), which stores the prognosed durations per resource-activity combination. consequently, the link between the metrices can be interchanged (distribution fitting, ...)

distribution fitting:
-create an additional matrix: r_a_matrix_fitted_objects, which stores the distribution objects
-the method .compute_percentiles() recomputes the percentiles of the fitted objects

example: 
  distribution fitting:
plan_object = Planning(historic_data=sample_intern, schedule=schedule_intern,
                       safety_percent=percentiles_intern[0], method='dist fitting')
plan_object.start_pipeline()
-> plan_object.schedule_updated_ends stores the revised schedules (new ends)
  
  standard deviations (mean based):
plan_object = Planning(historic_data=log_sized, schedule=log_intern.tail(50),
                       n_of_std_s=1, method='standard deviations')
plan_object.calculate_std_s(measure='mean')
-> plan_object.schedule_updated_ends stores the revised schedules (new ends)
