import os

def prepare_folder(train_id):
    os.system(f"cd results && mkdir {train_id}")
    os.system(f"cd results/{train_id}/ && mkdir dataset")
    os.system(f"cd results/{train_id}/ && mkdir models")
    os.system(f"cd results/{train_id}/ && mkdir figures")