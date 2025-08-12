import os

from ice.demo.client.report_file import ReportFile
from ice.demo.client.report_stats import ReportStats
from ice.demo.client.survey import Survey as ClientSurveyHandler
from ice.demo.server.survey import Survey as ServerSurveyHandler
from ice.job import IceJob, ServerConfig
from ice.server.relay import Relay

job_name = "ice_test"
job = IceJob(
    name=job_name,
    server_config=ServerConfig(config_data={"foo": 1, "bar": 2}),
)
job.set_app_packages(["ice"])

job.add_server_handler(ServerSurveyHandler())
job.add_server_handler(Relay("report.*", request_timeout=10.0))

job.add_client_handler(ClientSurveyHandler())
job.add_client_handler(ReportFile())
job.add_client_handler(ReportStats())

job_root = "/Users/yanc/NVFlare/sandbox/v27/prod_00/admin@nvidia.com/transfer"
job.export_job(job_root)
print(f"job created at {os.path.join(job_root, job_name)}")
