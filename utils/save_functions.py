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

def save_results_to_json(results, config):
    save_dir = config['eval_path']
    # 确保保存结果的目录存在
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # 定义文件名，包含实验条件
    filename = f"results_{config['model_type']}_b{config['batch_size']}_lr{config['learning_rate']}_len{config['max_length']}_nlayer{config['num_layers']}.json"
    # 完整的文件路径
    filepath = os.path.join(save_dir, filename)
    # 将results和config一起保存
    data_to_save = {
        'config': config,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)