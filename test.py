from config import Config
def return_str(config):
    save_file_header = config["model_type"]+"_b_"+str(config["batch_size"]) + "_lr_" + str(Config["learning_rate"])
    return save_file_header
header = return_str(Config)
print(header)  