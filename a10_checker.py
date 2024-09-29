import requests
import time
from pync import Notifier

API_KEY = 'REDACTED'

def check_a10_availability():
    url = "https://cloud.lambdalabs.com/api/v1/instance-types"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            instance_types = data['data']

            for instance_key, instance_info in instance_types.items():
                gpu_info = instance_info['instance_type']
                if 'gpu_description' in gpu_info and gpu_info['gpu_description'].strip() == "A10 (24 GB PCIe)":
                    regions = instance_info.get('regions_with_capacity_available', [])
                    if regions:
                        region_list = [region['description'] for region in regions]
                        if "Virginia, USA" in region_list:
                            print(f"A10 GPU is available in Virginia, USA!")
                            Notifier.notify("A10 GPU is available in Virginia, USA!", title="A10 GPU Available!")
                    else:
                        print("A10 GPU is currently unavailable.")
                    return
            print("No A10 GPU instance found in the data.")
        else:
            error_message = f"Error: Unable to access the API, status code {response.status_code}, message: {response.text}"
            print(error_message)
            Notifier.notify(error_message, title="API Access Error")
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        Notifier.notify(error_message, title="Script Error")

if __name__ == "__main__":
    while True:
        check_a10_availability()
        print("Checked for availability. Will check again in 2 minutes.")
        time.sleep(120)