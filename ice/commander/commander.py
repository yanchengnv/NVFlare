import os
import time
from typing import Dict, Optional

from ice.defs import PropKey, StatusCode
from ice.utils import download_files, make_file_refs_for_cell
from nvflare.fuel.f3.streaming.file_downloader import FileDownloader
from nvflare.fuel.flare_api.flare_api import new_secure_session


class Commander:

    def __init__(self, startup_kit_dir: str, user_name: str, login_timeout=5.0):
        self.sess = new_secure_session(startup_kit_location=startup_kit_dir, timeout=login_timeout, username=user_name)
        self.job_id = None
        self.cell = self.sess.api.get_cell()
        self.fqcn = self.cell.get_fqcn()

    def start(self, job_id: str = None, job_def_dir: str = None, timeout=10.0):
        """Start the admin session with either an existing job or a new job.

        Args:
            job_id: the id of the existing job. None if new job.
            job_def_dir: location of job definition. Not used if job_id is provided.
            timeout: how long to wait for the job to become ready.

        Returns: job id

        """
        if self.job_id:
            raise RuntimeError(f"this session is already running on job {job_id}")

        if not job_id and not job_def_dir:
            raise ValueError("either job_id or job_def_dir must be provided")

        if job_def_dir and not os.path.isdir(job_def_dir):
            raise ValueError(f"{job_def_dir} is not a valid dir")

        if not job_id:
            # try to submit the job
            job_id = self.sess.submit_job(job_def_dir)

            # wait until job is running
            start_time = time.time()
            while True:
                status = self.sess.get_job_status(job_id)
                print(f"job status: {status}")
                if status == "RUNNING":
                    break
                if time.time() - start_time > timeout:
                    raise RuntimeError("job is not started")
                time.sleep(1.0)

        # prob until job is ready
        start_time = time.time()
        while True:
            try:
                result = self.sess.do_app_command(job_id, topic="hello", cmd_data=None)
                if result.get(PropKey.STATUS) == StatusCode.OK:
                    break
            except:
                pass
            if time.time() - start_time > timeout:
                raise RuntimeError("job is not ready")
            time.sleep(1.0)

        self.job_id = job_id
        return job_id

    def send_command(
        self,
        topic: str,
        data: Optional[dict],
        timeout: float = 10.0,
        files: Optional[Dict[str, str]] = None,
        file_timeout_cb=None,
        **cb_kwargs,
    ) -> dict:
        """Send a command to the job

        Args:
            topic: topic of the command
            data: command data
            timeout: how long to wait for response
            files: files to be attached. Must be a dict of: file base name => location
            file_timeout_cb: called when timed out with the file names. You can use this CB to delete these files
                if necessary.

        Returns: result

        """
        if not self.job_id:
            raise RuntimeError("no job")

        file_refs = {}
        if files:
            file_refs = make_file_refs_for_cell(files, self.sess.api.cell, timeout, file_timeout_cb, **cb_kwargs)

        request = {
            PropKey.DATA: data,
            PropKey.ATTACHMENTS: file_refs,
        }

        self.sess.set_timeout(timeout)
        result = self.sess.do_app_command(self.job_id, topic=topic, cmd_data=request)

        # do we have any files to download?
        download_files(self.cell, result, timeout)
        return result

    def close(self):
        FileDownloader.shutdown()
        if self.sess:
            self.sess.close()
