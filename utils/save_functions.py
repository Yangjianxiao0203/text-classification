import json
import os
def save_as_json(path,file_name,data):
    save_path = os.path.join(path,file_name)
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if the data can be serialized as JSON
    try:
        serialized_data = json.dumps(data)
    except TypeError:
        raise ValueError("Provided data is not JSON serializable")

    with open(save_path, 'w') as file:
        file.write(serialized_data)