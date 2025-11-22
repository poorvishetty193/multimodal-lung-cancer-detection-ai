import requests

def submit_job(file_path, metadata_json):
    url = "http://localhost:8000/api/v1/scan/submit"
    files = {'file': open(file_path, 'rb')}
    data = {'metadata': metadata_json}
    response = requests.post(url, files=files, data=data)
    return response.json()

def get_job_status(job_id):
    url = f"http://localhost:8000/api/v1/job/{job_id}"
    response = requests.get(url)
    return response.json()

def get_job_results(job_id):
    url = f"http://localhost:8000/api/v1/job/{job_id}/results"
    response = requests.get(url)
    return response.json()

if __name__ == "__main__":
    file_path = "ui/src/pages/Status.jsx"  # sample file for testing
    metadata = "{}"  # empty metadata

    print("Submitting job...")
    submit_response = submit_job(file_path, metadata)
    print("Job submit response:", submit_response)

    job_id = submit_response.get("job_id")
    if not job_id:
        print("No job_id returned; aborting.")
        exit(1)

    import time
    print(f"Polling status for job_id: {job_id}")
    while True:
        status = get_job_status(job_id)
        print("Job status:", status)
        if status.get("status") in ["completed", "failed"]:
            break
        time.sleep(2)

    print("Fetching job results...")
    results = get_job_results(job_id)
    print("Job results:", results)
