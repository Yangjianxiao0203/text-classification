from config import Config
from utils.save_functions import save_all_json_to_csv

save_all_json_to_csv(Config["eval_path"], "result.csv")