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

after inserting sample data and a schedule, it re-computes the end timestamps of the schedules based on the sample data. it uses a resource-activity matrix/dataframe to store the raw values of durations per resource and activity pair (-> r_a_matrix_raw). by applying a function on the data (e.g. distribution fitting and computation of percentiles), it computes a second resource-activity matrix (-> r_a_matrix_values), which stores the prognosed durations per resource-activity combination.
