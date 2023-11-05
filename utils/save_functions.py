import json
import os
import csv
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
    return

def save_all_json_to_csv(output_dir, output_csv):
    # 获取output文件夹中所有的json文件
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

    # 创建或打开一个csv文件，准备写入数据
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(['model_type', 'epoch', 'num_layers', 'max_length', 'hidden_size',
                         'batch_size', 'learning_rate', 'acc', 'f1_score', 'recall', 'precision'])

        # 遍历每个json文件
        for json_file in json_files:
            with open(f'{output_dir}/{json_file}', 'r', encoding='utf-8') as j_file:
                data = json.load(j_file)
                try:
                    config = data['config']
                    results = data['results']
                except (KeyError, TypeError):
                    print(f"Error processing file: {json_file}")
                    continue

                # 提取所需的参数
                model_type = config['model_type']
                epoch = config['epoch']
                num_layers = config['num_layers']
                max_length = config['max_length']
                hidden_size = config['hidden_size']
                batch_size = config['batch_size']
                learning_rate = config['learning_rate']
                acc = results['accuracy']
                f1_score = results['f1_score']
                recall = results['recall']
                precision = results['precision']

                # 写入csv文件
                writer.writerow([model_type, epoch, num_layers, max_length, hidden_size,
                                 batch_size, learning_rate, acc, f1_score, recall, precision])
    return

def save_results_to_csv(results, config):
    output_csv = config['output_csv']
    # 提取所需的参数
    model_type = config['model_type']
    epoch = config['epoch']
    num_layers = config['num_layers']
    max_length = config['max_length']
    hidden_size = config['hidden_size']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    acc = results['accuracy']
    f1_score = results['f1_score']
    recall = results['recall']
    precision = results['precision']

    # 创建或打开一个csv文件，准备写入数据
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 检查文件是否为空，如果为空，写入标题行
        if file.tell() == 0:
            writer.writerow(['model_type', 'epoch', 'num_layers', 'max_length', 'hidden_size',
                             'batch_size', 'learning_rate', 'acc', 'f1_score', 'recall', 'precision'])
        # 写入数据行
        writer.writerow([model_type, epoch, num_layers, max_length, hidden_size,
                         batch_size, learning_rate, acc, f1_score, recall, precision])


def save_lora_results_to_csv(results,config):
    output_lora_csv = config['output_lora_csv']
    # 提取所需的参数
    lora_r = config['lora_r']
    lora_alpha = config['lora_alpha']

    acc = results['accuracy']
    f1_score = results['f1_score']
    recall = results['recall']
    precision = results['precision']

    with open(output_lora_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 检查文件是否为空，如果为空，写入标题行
        if file.tell() == 0:
            writer.writerow(['lora_r','lora_alpha','acc', 'f1_score', 'recall', 'precision'])
        # 写入数据行
        writer.writerow([lora_r,lora_alpha,acc, f1_score, recall, precision])
    return





