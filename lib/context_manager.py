import datetime
import os
import pandas as pd
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from time import sleep

class DirNotEmpty(Exception):
    pass

@dataclass
class SbatchManager:
    filename: str
    slurmUser: str
    userQueue: pd.DataFrame

    @classmethod
    def setup(cls, filename):
        slurmUser = os.popen("whoami").read()
        slurmUser.strip()
        userQueue = os.popen(f"squeue -u {slurmUser}").read()
        userQueue = pd.DataFrame([x.split() for x in userQueue.split("\n")]).dropna()
        userQueue.columns = userQueue.iloc[0]
        userQueue = userQueue[1:]
        jobdata = [os.popen(f"scontrol show jobid -dd {i}").read() for i in userQueue["JOBID"]]
        workdirs = []
        # TODO works, just ugly
        for output in jobdata:
            lines = output.split("\n")
            for line in lines:
                # a bit hacky
                if "WorkDir" in line[:11]:
                    print(line)
                    workdirs.append("/" + "/".join(line.split("/")[1:]))
        userQueue["jobData"] = jobdata
        userQueue["workDir"] = workdirs
        return cls(filename, slurmUser, userQueue)

    def safety(self):
        cwd = os.getcwd()
        if cwd in self.userQueue["workDir"]:
            raise Exception("A job is currently running from this directory!")

    def launch(self):
        self.safety()
        if os.path.exists("submitted"):
            raise Exception("job already run from this directory")
        printout = os.popen(f"sbatch {self.filename} && touch submitted").read()
        with open("submitted", "w") as f:
            f.write(str(datetime.datetime.now()))
        print(printout)
        self.jobId = printout.split()[-1]

    def wait_for_job(self, extra_time=4):
        # Waits for a job to finish. Consider setting sbatch dependencies instead.
        done = False
        while not done:
            if os.path.exists("tc.out") or os.path.exists(f"slurm-{self.jobId}.out"):
                done = True
            else:
                sleep(5)
        print("Found output!")
        sleep(extra_time)

@contextmanager
def minimal_context(newDir, tc_input, sbatch_input):
    origDir = os.getcwd()
    print(newDir)
    if newDir.exists():
        raise DirNotEmpty
    else:
        newDir.mkdir(parents=True)
    try:
        os.chdir(os.path.expanduser(newDir))
        shutil.copy(tc_input, newDir)
        shutil.copy(sbatch_input, newDir)
        yield SbatchManager.setup("sbatch.sh")
        #yield RunManager.create_manager(tc_file_name, sbatch_file_name, extra_files, debug)
    finally:
        os.chdir(origDir)

@contextmanager
def enter_dir(dir):
    origDir = os.getcwd()
    if not dir.exists():
        raise Exception("directory does not exits")
    os.chdir(os.path.expanduser(dir))
    try:
        yield
    finally:
        os.chdir(origDir)