import os
import time

from ice.commander.commander import Commander

user_name = "admin@nvidia.com"
startup_dir = f"/Users/yanc/NVFlare/sandbox/v27/prod_00/{user_name}"

# create the Commander object.
commander = Commander(startup_dir, user_name)

# start the long-running job.
# You either start the job from scratch; or you can use the existing job ID if it's already running.
# If you start a new job, it may take a few seconds for the job to become ready. The "start" call won't return
# until the job is ready.
job_def = os.path.join(startup_dir, "transfer", "ice_test")
job_id = commander.start(job_def_dir=job_def)
# job_id = commander.start(job_id="d63d6dfd-b353-4cb8-ae71-f0c0e978b92d")

# Now the job is ready, you can send commands to it.
start_time = time.time()
result = commander.send_command("survey", data=None)
print(f"survey result [{time.time()-start_time}]: {result}")

start_time = time.time()
result = commander.send_command("xyz", data=None)
print(f"xyz result [{time.time()-start_time}]: {result}")

# send a command with attached files.
# Files can be anything, even python code. But be mindful about security!
file_dir = "/Users/yanc/NVFlare/ice/demo"

start_time = time.time()
result = commander.send_command(
    "report.file",
    data=None,
    files={"driver.py": os.path.join(file_dir, "driver.py"), "make_job.py": os.path.join(file_dir, "make_job.py")},
)
print(f"report.file result [{time.time()-start_time}]: {result}")

# query for stats
start_time = time.time()
result = commander.send_command("report.stats", data=None)
print(f"report.stats result [{time.time()-start_time}]: {result}")

# Stop the job if you are done. BUT don't do this if the job needs to keep running.
commander.stop_job()

# Close the session.
commander.close()
print(f"Done with job {job_id}")
