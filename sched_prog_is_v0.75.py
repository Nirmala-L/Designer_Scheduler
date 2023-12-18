# +++++++++++++++++++++++++++++++++++++++++++++++++++
# scheduling_program v0.73
# coding started on 20231109
# Revision 20231128
# Revision notes for 0.73
#      Fixed the problem with low skilled workers being overlooked for jobs that they can do
#      Now the initial solution program result reflects the proposed initial solution
# Implemented CSV reading for job/worker data
# +++++++++++++++++++++++++++++++++++++++++++++++++++


import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
# import time


# Function to convert CSV files containing NAN values
def convert_csv_to_list(filename):
    # open file containing data
    dataframe = pd.read_csv(filename,
                            index_col=False)
    # Create empty output list
    out_list = []
    dataframe = dataframe.drop(columns=dataframe.columns[:1])
    value_set = dataframe.values.tolist()
    item_count = len(value_set)
    i = 0

    for i in range(item_count):
        line = value_set[i]
        temp_list = [element for element in line if not math.isnan(element)]
        out_list.append(temp_list)

    return out_list


# Define time related variables
def_year = 2023  # define year
def_month = 12  # define month

period_start = datetime(year=def_year,
                        month=def_month,
                        day=1,
                        hour=9,
                        minute=0
                        )

# Enumerator list for jobs
jobs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# 1.1 Job due dates
dueDate = [1000, 18, 15, 19, 14, 10, 11, 25, 20, 27, 29, 15, 21]
dueDate_datetime = []
for member in range(len(jobs)):
    member_duedate = period_start + timedelta(days=dueDate[member])
    dueDate_datetime.append(member_duedate)

# 1.2 Job due dates
# tardiness = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tardiness = []
for i in range(len(jobs)):
    tardiness.append(timedelta(0))

# 2. Operations
# 2.1 Operation standard times
operations = [[0, 0, 0, 0],
              [0, 1, 2, 1, 2, 4, 3, 3],
              [0, 2, 2, 2, 1, 6, 5, 3],
              [0, 1, 1, 2, 2, 1, 3, 3, 3, 1, 4, 5, 5, 3],
              [0, 1, 1, 1, 2, 2, 1, 3, 2, 1, 1],
              [0, 2, 1, 2, 1, 3, 1, 1, 1, 3, 3, 1, 1, 4, 3, 2, 3, 1],
              [0, 1, 1, 1, 2, 1, 1, 2, 2, 3, 1],
              [0, 1, 2, 3, 1, 2, 1, 1, 2, 1],
              [0, 3, 2, 3, 1, 1, 2, 2, 3, 2, 4, 2, 1, 3, 2, 3, 4, 6, 3, 5, 3, 3],
              [0, 1, 1, 1, 2, 2, 1, 2],
              [0, 2, 3, 2, 2, 2, 1, 3, 1, 1, 4, 5, 3, 1],
              [0, 1, 1, 2, 1, 2, 1, 3, 2, 3, 1],
              [0, 3, 2, 3, 3, 3, 4, 3, 2, 3, 1, 3, 3, 5, 4, 5, 3, 7, 5, 5, 4, 3]
              ]
# Major issue - how to implement precedence constraints
preced_const = []

# 2.2 Operation skill limits
sk_lim = [[0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.1, 0.2, 0.4, 0.4],
          [0.0, 0.2, 0.2, 0.6, 0.7],
          [0.0, 0.4, 0.6, 0.7, 0.8],
          [0.0, 0.1, 0.3, 0.4, 0.4],
          [0.0, 0.2, 0.5, 0.7, 0.7],
          [0.0, 0.1, 0.2, 0.4, 0.5],
          [0.0, 0.2, 0.2, 0.3, 0.4],
          [0.0, 0.5, 0.7, 0.9, 0.9],
          [0.0, 0.1, 0.1, 0.2, 0.3],
          [0.0, 0.4, 0.4, 0.6, 0.7],
          [0.0, 0.1, 0.2, 0.4, 0.4],
          [0.0, 0.6, 0.8, 0.9, 0.9]
          ]

# 2.3 Operation task types
task_type = [[0, 0, 0, 0],
             [0, 1, 1, 2, 2, 3, 4, 4],
             [0, 1, 1, 2, 2, 3, 4, 4],
             [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4],
             [0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4],
             [0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4],
             [0, 1, 1, 1, 1, 2, 2, 3, 4, 4],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4],
             [0, 1, 1, 2, 2, 3, 4, 4],
             [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4],
             [0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4]
             ]

# Job operation completion registers
task_finish_filename = "task_finish.csv"
task_finish = convert_csv_to_list(task_finish_filename)
print(task_finish)
task_finish_time_filename = "task_finish_time.csv"

task_finish_time = convert_csv_to_list(task_finish_time_filename)

# **************WORKERS************

workers = [0, 1, 2, 3, 4]

num_workers = len(workers)

# 1.1 Worker categories
w_cats = [0, 1, 2, 3, 4]

# Worker data - Read from csv

df_skill = pd.read_csv("Worker_data_task_skill.csv",
                       index_col=False)

df_count = pd.read_csv("Worker_data_task_count.csv",
                       index_col=False)

df_count = df_count.drop(columns=df_count.columns[:1])
df_skill = df_skill.drop(columns=df_skill.columns[:1])
# 2. Worker skill limits

"""w_skill = [[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1],
           [0, 0.7029, 0.7049, 0.7044, 0.7078],
           [0, 0.5016, 0.5016, 0.5013, 0.5023],
           [0, 0.2013, 0.2, 0.2, 0.2]
           ]
        """
w_skill = df_skill.values.tolist()


# 2.1 Worker task counts
"""
w_task_count = [[0, 0, 0, 0, 0],
                [0, 28, 12, 4, 12],
                [0, 25, 21, 8, 21],
                [0, 5, 5, 3, 5],
                [0, 9, 1, 1, 1]]
"""

w_task_count = df_count.values.tolist()
# 3. Worker maximum times
w_max_time = [0, 100, 120, 160, 160]

# 4. Worker overtime costs
w_ot_cost = [0, 1, 0.5, 0.3, 0.1]
# 5. Worker OT hours
w_ot_time = [0, 0, 0, 0, 0]

# **********************

# Verifications
num_jobs = len(jobs)
if num_jobs != len(dueDate):
    print("Error, mismatch in data - additional or missing data in due date")
elif num_jobs != len(operations) or num_jobs != len(task_type):
    print("Error, mismatch in data - skill limit or task type number doesn't match with job data")

for i in range(num_jobs):
    if len(operations[i]) != len(task_type[i]):
        print("Error! Mismatch in job ", i + 1, " data\nPlease verify!")


def calculate_finish_time(start_time, processing_time, overtime_ok, worker):

    finish_time = start_time + timedelta(hours=processing_time)
    cat = w_cats[worker]
    # st_time = 17
    st_time_dic = {1: 14,
                   2: 15,
                   3: 17,
                   4: 17}
    st_time = st_time_dic[cat]

    gap = st_time - start_time.hour
    if gap < processing_time:
        carryover = processing_time - gap
        if overtime_ok == 1:
            carryover = carryover-2
            w_ot_time[worker] += carryover
            if carryover > 0:
                finish_time = start_time + timedelta(days=1)
                finish_time = finish_time.replace(hour=9)
                finish_time = finish_time + timedelta(hours=carryover)
            else:
                finish_time = start_time + timedelta(hours=processing_time)
        else:
            finish_time = start_time + timedelta(days=1)
            finish_time = finish_time.replace(hour=9)
            finish_time = finish_time + timedelta(hours=carryover)

    return finish_time


# shift dates by 1 when reaching the end of day


# Determine number of operations
# List to store count of operations
j_opCount = []

# loop to register the actual count of operations
for i in range(num_jobs):
    j_opCount.append(len(operations[i]))


def st_time(job, operation):
    return operations[job][operation]


def op_t_type(job, operation):
    out_ttype = task_type[job][operation]
    # print(out_ttype)
    return out_ttype


def skill_limit(job, operation):
    task = op_t_type(job, operation)
    out_skill_limit = sk_lim[job][task]
    # print(out_skill_limit)
    return out_skill_limit


def capacity(worker, op_ttype):
    out_capacity = w_skill[worker][op_ttype]
    # print(out_capacity)
    return out_capacity


def let_pair_alloc(worker, job, operation):
    return True


def permit(worker, job, operation):
    task_cat = op_t_type(job, operation)
    if capacity(worker, task_cat) >= skill_limit(job, operation):
        return 1
    # elif let_pair_alloc(worker, job, operation):
    #    return 1
    else:
        return -1


def processing_time(worker, job, operation):
    if permit(worker, job, operation) == 1 and capacity(worker, op_t_type(job, operation)) > 0:
        process_time = st_time(job, operation) / capacity(worker, op_t_type(job, operation))
        process_time_int = math.ceil(process_time)
        return process_time_int
    elif permit(worker, job, operation) == -1:
        print("Worker", worker, "cannot process the job")
        return -1


def worker_finish_time(job, operation, worker):
    proc_time = processing_time(job, operation, worker)
    earliest_start = end_times_workers[worker]
    earliest_fin_time = calculate_finish_time(earliest_start, proc_time, ot_allowed, worker)
    return earliest_fin_time


def compare_finish_time(job, operation, worker1, worker2):
    fin_time1 = worker_finish_time(job, operation, worker1)
    fin_time2 = worker_finish_time(job, operation, worker2)
    if fin_time1 < fin_time2:
        outworker = worker1
    else:
        outworker = worker2

    return outworker


# Change the date of operation
def shift_date(datevar):
    tempVar = datevar
    allowed_time = 17
    # cat = w_cats[worker]

    if (type(datevar) != datetime):
        print("datatype mismatch")

    if tempVar.hour >= allowed_time:
        # print(1)
        tempVar = tempVar + timedelta(days=1)
        tempVar = tempVar.replace(hour=9)
        print(tempVar)

    return tempVar


due_date = pd.DataFrame({"Jobs": jobs,
                         "Due Dates": dueDate}
                        )


# temp file to check finish times of workers
finish_lists = [[], [], [], [], []]


# sort jobs by the due date [^_^]
# does not consider tiebreak - tiebreak simply implemented through job number
def create_priority_list_edd():
    new_order_list = due_date.sort_values(by="Due Dates")
    new_job_list = new_order_list["Jobs"].values.tolist()
    return new_job_list

# todo Change these into dynamic allocations
start_times_workers = [period_start,
                       period_start,
                       period_start,
                       period_start,
                       period_start
                       ]

earliest_start_times_workers = [period_start,
                                period_start,
                                period_start,
                                period_start,
                                period_start
                                ]

end_times_workers = [period_start,
                     period_start,
                     period_start,
                     period_start,
                     period_start
                     ]

start_time_jobs = [period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start,
                   period_start
                   ]

work_time_workers = [0, 0, 0, 0, 0]
end_time_jobs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
end_time_jobs_days = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# skill improvement variables
total_skill_imp = 0
ro = 0.01
phi = 0.005


# ******************INITIAL SOLUTION GENERATION BEGINS HERE******************

# Selection function version 1
def select_better(worker1, worker2, job, operation):
    time_worker1 = processing_time(worker1, job, operation)
    time_worker2 = processing_time(worker2, job, operation)
    best_time = max(time_worker1, time_worker2)
    if best_time == time_worker1:
        return worker1
    elif best_time == time_worker2:
        return worker2


# Selection function version 2 - Get worker from a list
def select_better_worker(workers_list, job, operation):
    time_worker_best = 0
    worker_index = 0
    best_worker_index = 0
    for worker in workers_list:
        worker_index = worker
        process_time = processing_time(worker, job, operation)
        if process_time > time_worker_best:
            best_worker_index = worker_index
            time_worker_best = process_time

    return best_worker_index

# ***************IMPORTANT****************
# Set OT allowance
ot_allowed = 1
# ---------------------------------------

new_order = create_priority_list_edd()
new_using_order = []
for job in new_order:
    new_using_order.append(job)
print("This is the new job priority order", new_using_order)

alloc_list = []
job_alloc_list = []
prec_con = []
worker_list=[]
total_work_time= 0

# dynamically initialize the precedence constraints matrix
for i in range(num_jobs+1):
    prec_con.append(0)


for i in range(num_jobs):
    prec_df = pd.read_excel("precedence_constraints.xlsx", sheet_name=i)
    # print(i)                              # testing
    prec_con[i] = prec_df.values.tolist()
    # print(prec_con[i])                    # testing

# for loop to allocate workers
for job in new_using_order:
    if job == 0:
        continue
    workers_for_job = []
    job_worker_list = []
    print("THis job is ", job)
    print("This job has ", len(operations[job]), "operations")
    i = 1
    start_time_ops = []
    end_time_ops = [0]

    while i < j_opCount[job]:
        index_of_last_operation = j_opCount[job] - 1
        print(job, ",", i)
        op_task_type = op_t_type(job, i)
        print("The task type is ", op_task_type)

        # print(select_better(2,3,4,1))
        # last_worker = workers_for_job[-1]
        if not workers_for_job:
            final_suitable_worker = select_better_worker(workers, job, i)

        elif len(workers_for_job) > 1 and (workers_for_job[-1] == workers_for_job[-2]):
            last_worker = workers_for_job[-1]
            if op_task_type == task_type[job][i-1] and processing_time(last_worker, job, i) > 0:
                final_suitable_worker = last_worker
            else:

                final_suitable_worker = select_better_worker(workers, job, index_of_last_operation)
                           
        else:
            last_worker = workers_for_job[-1]
            final_suitable_worker = last_worker

        workers_for_job.append(final_suitable_worker)
        print("The selected worker for job ", job, " operation ", i, "is", final_suitable_worker)

        # increase task count
        w_task_count[final_suitable_worker][op_task_type] += 1
        w_t_count = w_task_count[final_suitable_worker][op_task_type]

        # skill improvement calculations
        skill_imp = phi*skill_limit(job, i) / w_t_count
        new_skill = capacity(final_suitable_worker, op_task_type) + skill_imp
        if capacity(final_suitable_worker, op_task_type) < 1:
            w_skill[final_suitable_worker][op_task_type] = new_skill
            total_skill_imp += skill_imp
            print(skill_imp, total_skill_imp)

        # Precedence calculations
        earliest_start_next = period_start
        precs = len(prec_con[job][i])
        print('Precs= ', precs)
        prec_ops = prec_con[job][i]
        for k in range(precs):
            # print(k)
            x = prec_ops[k]
            preceding_ops = []
            if not np.isnan(x) and x !=0:
                print("This operation has the following preceding operations :", x)
                preceding_ops.append(int(x))

                finish_time_latest = end_time_ops[int(x)]
                if earliest_start_next < finish_time_latest:
                    earliest_start_next = finish_time_latest
                print("Finish time of that operation is", end_time_ops[preceding_ops[-1]])
                # print(x)

        available_time = earliest_start_next
        print("The worker can process job from :", available_time)
        precs_index = len(preceding_ops)
        for a in range(precs_index):
            prec_index = preceding_ops[a]
            print("the preceding operation", prec_index, "finishes at", end_time_ops[prec_index] )
            if prec_index != 0 and available_time < end_time_ops[prec_index]:
                print("available time changed to")
                available_time = end_time_ops[prec_index]
                print(available_time)

        # Calculate the start times and end times
        # print(available_time)
        # print(end_times_workers[final_suitable_worker])

        start_time = max(available_time, end_times_workers[final_suitable_worker])
        print("the operation starts at", start_time)
        act_proc_time = processing_time(final_suitable_worker, job, i)
        end_time_tstamp = calculate_finish_time(start_time, act_proc_time, ot_allowed, final_suitable_worker)
        # end_time_op = start_time + act_proc_time

        # Add the finish time to worker
        # Verify if the time has exceeded the maximum work hour for the employee
        if end_time_tstamp.hour >= 17:
            end_times_workers[final_suitable_worker] = shift_date(end_time_tstamp)
        else:
            end_times_workers[final_suitable_worker] = end_time_tstamp
        finish_lists[final_suitable_worker].append(end_times_workers[final_suitable_worker])
        # Record start time
        start_time_ops.append(start_time)
        # Record end time
        end_time_ops.append(end_time_tstamp)
        task_finish_time[job][i] = end_time_tstamp

        # Append the necessary data for the operation list
        job_alloc_list.append(job)  # Job Number
        job_alloc_list.append(i)  # Operation Number
        job_alloc_list.append(op_task_type)
        job_alloc_list.append(st_time(job, i))
        job_alloc_list.append(final_suitable_worker)  # Allocated worker
        job_alloc_list.append(act_proc_time)  # Time taken to process
        job_alloc_list.append(start_time)  # Start time
        job_alloc_list.append(end_time_tstamp)  # End time

        # printing confirmation
        print("The processing time is ", act_proc_time)
        total_work_time += act_proc_time
        work_time_workers[final_suitable_worker] += act_proc_time

        # Append the operation list to the job info list
        alloc_list.append(job_alloc_list)
        job_worker_list.append(final_suitable_worker)

        # change operation finish register value to 1
        task_finish[job][i] = 1

        # add the worker to the worker list

        job_alloc_list = []  # clear the operation list
        if i == index_of_last_operation:
            end_time_jobs[job] = end_time_tstamp
        i += 1
# **************  END OF LOOP  *************
    worker_list.append(job_worker_list)

    if job == new_using_order[-2]:
        print("Finished allocating all jobs\nComputing results")
    else:
        print("Loading next_job......")

    finished_day = end_time_jobs[job].day
    end_time_jobs_days[job] = finished_day

    workers_for_job.clear()
    finished_day = 0

# Convert the job info list to a dataframe
df_Allocation = pd.DataFrame(alloc_list, columns=["Job",
                                                  "Operation",
                                                  "Task type",
                                                  "Standard time",
                                                  "Allocated Worker",
                                                  "Processing_time",
                                                  "Start_time",
                                                  "End_time"
                                                  ])
# Write the file into CSV
df_Allocation.to_csv("df_Allocation_pyCharm_iterated.csv", index=True)

# Attempt to save nested list with uneven lengths into a CSV
df_worker_list = pd.DataFrame(worker_list, columns=["TAlloc1",
                                                    "TAlloc2",
                                                    "TAlloc3", "TAlloc4",
                                                    "TAlloc5", "TAlloc6",
                                                    "TAlloc7", "TAlloc8",
                                                    "TAlloc9", "TAlloc10",
                                                    "TAlloc11", "TAlloc12",
                                                    "TAlloc13", "TAlloc13",
                                                    "TAlloc15", "TAlloc16",
                                                    "TAlloc17", "TAlloc18",
                                                    "TAlloc19", "TAlloc20",
                                                    "TAlloc21"
                                                    ])
df_worker_list.to_csv("Job_Operation_Allocation_List.csv")

print("The end times for workers in hours: ", end_times_workers)
print()


# Objective value 1 calculation
# Variable to store OT cost
ot_cost = 0
i = 1
# ----------------------------------------

# Iterate the ot costs list to find total ot cost
for i in range(len(w_ot_cost)):
    # ot_cost += w_ot_cost[i]*end_times_workers[i]
    ot_cost += w_ot_cost[i] * w_ot_time[i]

j=1
total_tardiness = 0
for j in range(len(jobs)):
    #tardiness_job = end_time_jobs_days[j] - dueDate[j]
    if j == 0:
        continue
    tardiness_job = end_time_jobs[j] - dueDate_datetime[j]
    print(tardiness_job)
    tardiness[j] = max(tardiness_job, timedelta(days=0))
    tardiness_val = tardiness[j].total_seconds()
    tardiness_val = math.ceil(tardiness_val/(24*60*60))
    total_tardiness += tardiness_val

print("The end dates of the jobs: ", end_time_jobs_days)
print("The job tardiness is : ")
print(tardiness)
print(total_tardiness)


# Objective function calculation
# Define arbitrary constant
tard_const = 4

# calculate weighted tardiness and objective function
weighted_tard = tard_const*total_tardiness
Obj_func = ot_cost + weighted_tard
i=1

for i in range(num_workers):
    df_finish_times = pd.DataFrame(finish_lists[i])
    with pd.ExcelWriter("Finish_times.xlsx") as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet

        df_finish_times.to_excel(writer, sheet_name=str(i), index=False)

# Output worker related data
df_w_skill = pd.DataFrame(w_skill, columns=["Task_0_Skill",
                                            "Task_1_Skill",
                                            "Task_2_Skill",
                                            "Task_3_Skill",
                                            "Task_4_Skill"
                                            ], index=workers)
df_w_task_count = pd.DataFrame(w_task_count, columns=["Task_0_Count",
                                                      "Task_1_Count",
                                                      "Task 2_Count",
                                                      "Task_3_Count",
                                                      "Task_4_Count"
                                                      ], index=workers)

with pd.ExcelWriter("Worker_data_iterated.xlsx") as writer:
    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet

    df_w_task_count.to_excel(writer, sheet_name="Task_Count", index=True)
    df_w_skill.to_excel(writer, sheet_name="Task_Skill", index=True)

# Calculate utilization of each worker
worker_util = []

for worker in workers:

    utilization = work_time_workers[worker]/total_work_time
    worker_util.append(utilization)
# calculate inverse of skill improvement
inverse_skill_imp = 1/total_skill_imp

# initiate the lists to hold results of iterations
iterated_ot_cost = []
iterated_total_tardiness = []
iterated_weighted_tardiness = []
iterated_total_skill_imp = []
iterated_inverse_total_skill_imp = []
iterated_total_work_time = []
iteration_summary = []

temp_list_store = []

# add the iteration results to the different lists
iterated_ot_cost.append(ot_cost)
temp_list_store.append(ot_cost)
iterated_total_tardiness.append(total_tardiness)
temp_list_store.append(total_tardiness)
iterated_weighted_tardiness.append(weighted_tard)
temp_list_store.append(weighted_tard)
iterated_total_skill_imp.append(total_skill_imp)
temp_list_store.append(total_skill_imp)
iterated_inverse_total_skill_imp.append(inverse_skill_imp)
temp_list_store.append(inverse_skill_imp)
iterated_total_work_time.append(total_work_time)
temp_list_store.append(total_work_time)
iteration_summary.append(temp_list_store)
temp_list_store = []


print("*******Results*********")
print("The OT cost is", ot_cost)
print("The tardiness is", total_tardiness)
print("The weighted tardiness is", weighted_tard)
print("The objective function value for this iteration is ", Obj_func)
print("The total skill improvement for this iteration is ", total_skill_imp)

print("The inverse of total skill improvement is:", float("{:.2f}".format(inverse_skill_imp)))
print("The new skill improvement matrix is")
for row in w_skill:
    print(row)
print("The new task count is")
print(w_task_count)
print("The total work hours accumulated = ", total_work_time)
print("working time for each worker: ")
print(work_time_workers)
print("Utilization rates for workers:")
print(worker_util)
print("\n", iteration_summary[0])
print("*************Iteration complete*************")
print("-------------------------------------------")
print("*****************************************")
print("---------------------------------------")
print("*************************************")


# Tabu search program
# Skill based improvements
# todo Create the program


# time-based improvements
# select the job with the worst tardiness
print("Attempting to improve the results for tardiness")

worst_tardiness = timedelta(0)
worst_tard_index = 0
for i in range(len(jobs)):
    if i ==0:
        continue
    if worst_tardiness < tardiness[i]:
        worst_tardiness = tardiness[i]
        worst_tard_index = i

print("The job with worst tardiness is", worst_tard_index)

# find the allocated workers for the job
allocated_workers_for_change = df_Allocation.loc[df_Allocation["Job"] == worst_tard_index,
                                                 ["Allocated Worker"]]

allocated_workers_for_change = allocated_workers_for_change.drop_duplicates()
allocated_workers_for_change = allocated_workers_for_change.values.tolist()







