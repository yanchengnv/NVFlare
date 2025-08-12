import os
import time

from ice.commander.commander import Commander

user_name = "admin@nvidia.com"
startup_dir = f"/Users/yanc/NVFlare/sandbox/v27/prod_00/{user_name}"

commander = Commander(startup_dir, user_name)

job_def = os.path.join(startup_dir, "transfer", "ice_test")
job_id = commander.start(job_def_dir=job_def)

start_time = time.time()
result = commander.send_command("survey", data=None)
print(f"survey result [{time.time()-start_time}]: {result}")

start_time = time.time()
result = commander.send_command("xyz", data=None)
print(f"xyz result [{time.time()-start_time}]: {result}")

# send files
file_dir = "/Users/yanc/NVFlare/ice/demo"

start_time = time.time()
result = commander.send_command(
    "report.file",
    data=None,
    files={"user.py": os.path.join(file_dir, "user.py"), "make_job.py": os.path.join(file_dir, "make_job.py")},
)
print(f"report.file result [{time.time()-start_time}]: {result}")

# query for stats
start_time = time.time()
result = commander.send_command("report.stats", data=None)
print(f"report.stats result [{time.time()-start_time}]: {result}")

# shutdown the job
commander.send_command("bye", data=None)
commander.close()
print(f"Done with job {job_id}")
