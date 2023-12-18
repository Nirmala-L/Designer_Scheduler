# Program to create Gantt chart from the
import plotly.express as px
import pandas as pd
from datetime import datetime

time = datetime.now()
# datafile = "df_Allocation_-EDD_2.csv"
# datafile = "df_Allocation_pyCharm_iterated_2.csv"
# datafile = "recalculated_schedule.csv"
datafile = "newfile.csv"
test_string = 'EDD'

if test_string in datafile:
    filename = "Gantt_chart_withoutOT_EDD"+str(time.strftime("%Y-%m-%d-%H-%M"))+".html"
    titlebar = "Gantt chart under EDD"
elif "calc" in datafile:
    filename = "Gantt_chart_rescheduled"+str(time.strftime("%Y-%m-%d-%H-%M"))+".html"
    titlebar = "Gantt chart under reschedule"
else:
    filename = "Gantt_chart_withOT_myMethod"+str(time.strftime("%Y-%m-%d-%H-%M"))+".html"
    titlebar = "Gantt chart under mymethod with OT"


# import result file
df = pd.read_csv(datafile)
# Arrange the jobs in the numerical order of identifiers
df = df.sort_values(by=["Job", "Operation"])

# Change format of
df["Job"] = df["Job"].astype(str)
df["Operation"] = df["Operation"].astype(str)
df["Allocated Worker"] = df["Allocated Worker"].astype(str)
df.to_csv("Gantt_check.csv", index=False)
fig_2 = px.timeline(df, x_start="Start_time", x_end="End_time", y="Job",
                    title=titlebar,
                    height=900, width=1600, # regular
                    # height=200, width=1200, # widescreen for PPT
                    text="Operation",
                    color="Allocated Worker")
fig_2.update_layout(legend_traceorder="reversed")
fig_2 = fig_2.update_yaxes(autorange="reversed")
# fig_2 = fig_2.update_xaxes(tickformat="%d\n%b", ticklabelmode="period")
fig_2 = fig_2.update_xaxes(minor=dict(ticks="outside", showgrid=True))
fig2 = fig_2.update_traces(width=1)

fig_2.show()


if test_string in datafile:
    filename = "Gantt_chart_withoutOT_EDD"+str(time.strftime("%Y-%m-%d-%H-%M"))+".html"
elif "calc" in datafile:
    filename = "Gantt_chart_rescheduled_"+str(time.strftime("%Y-%m-%d-%H-%M"))+".html"
else:
    filename = "Gantt_chart_withOT_myMethod"+str(time.strftime("%Y-%m-%d-%H-%M"))+".html"

fig_2.write_html(filename)
