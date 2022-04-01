import requests
import time
import os
import pprint

api_key = "f02aea90975d4349a92ae635214a5103"
endpoint = "https://api.bing.microsoft.com/"

url = f"{endpoint}/v7.0/images/search"

headers = { "Ocp-Apim-Subscription-Key": api_key }

params = {
    "q": "face with mask",
    "license": "public",
    "imageType": "photo",
    "safeSearch": "Strict",
}

response = requests.get(url, headers=headers, params=params)
response.raise_for_status()

result = response.json()

pprint.pprint(result)

new_offset = 0
contentUrls = []

new_offset = 0
contentUrls = []

while new_offset <= 50:
    # print(new_offset)
    params["offset"] = new_offset

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    result = response.json()

    time.sleep(1)

    new_offset = result["nextOffset"]

    for item in result["value"]:
        print(item["contentUrl"])
        contentUrls.append(item["contentUrl"])

dir_path = "./with_mask/"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for url in contentUrls:
    split = url.split("/")

    last_item = split[-1]

    second_split = last_item.split("?")

    if len(second_split) > 1:
        last_item = second_split[0]

    third_split = last_item.split("!")

    if len(third_split) > 1:
        last_item = third_split[0]

    print(last_item)
    path = os.path.join(dir_path, last_item)

    try:
        with open(path, "wb") as f:
            image_data = requests.get(url)
            # image_data.raise_for_status()

            f.write(image_data.content)
    except OSError:
        pass