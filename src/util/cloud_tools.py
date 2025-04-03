import datetime
import json
import os
import requests
import shutil
from util.logger import setup_logger


log = setup_logger()


def get_pod_and_terminate():
    """
    Sends a GET request to get the pod id and then terminates it to shutdown.
    """
    run_pod_token = os.getenv("RP_SECRET")
    response = requests.get("https://rest.runpod.io/v1/pods",
                            headers={
                                "Authorization": f"Bearer {run_pod_token}"
                            }
                            ).json()
    print("Received pod list response:")
    print(json.dumps(response, indent=2))
    # Response is a json array, ideally should have only one pod info
    pod = response[0]
    podId = pod["id"]
    response = requests.delete(f"https://rest.runpod.io/v1/pods/{podId}",
                               headers={
        "Authorization": f"Bearer {run_pod_token}"
    }
    )
    print("Pod termination response status:", response.status_code)
    print("Response body:", response.text)


def auto_shutdown(copy_dir: str = "/mnt/ai"):
    """Performs auto shutdown for cloud environment.

    This function copies the 'weights' and 'logs' directories to a new directory created as
    run_{timestamp} inside the specified copy_dir. The new folder will contain the contents
    of the original 'weights' and 'logs' directories.

    Args:
        copy_dir (str): The base directory where the 'weights' and 'logs' folders will be copied.
    """
    log.info("Running cloud shutdown script")

    # Create a new folder with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = os.path.join(copy_dir, "cv_coursework", f"run_{timestamp}")
    os.makedirs(new_folder, exist_ok=True)

    log.info(f"Saving backup to {new_folder}")

    # List of directories to copy (assumed to be in the current working directory)
    src_dirs = ["weights", "logs"]
    for src_dir in src_dirs:
        if os.path.exists(src_dir):
            dest_dir = os.path.join(new_folder, src_dir)
            shutil.copytree(src_dir, dest_dir)
            print(f"Copied {src_dir} to {dest_dir}")
        else:
            print(f"Source directory {src_dir} does not exist")

    get_pod_and_terminate()
