import boto3
import os

s3 = boto3.client('s3')

def upload_dataset():
    bucket_name = 'deeplearningbucket'
    s3_path = 'data/emotion_data'
    local_path = '../data'
    #upload every file in local_path to s3_path
    for root, dirs, files in os.walk(local_path):
        for file in files:
            s3_file_path = os.path.join(root.replace(local_path, s3_path), file)
            local_file_path = os.path.join(root, file)
            s3.upload_file(local_file_path, bucket_name, s3_file_path)
            print(s3_file_path)

    # s3.upload_file(local_path, bucket_name, s3_path)
def upload_bert():
    bucket_name = 'deeplearningbucket'
    s3_path = 'data/bert'
    local_path = '../bert-base-uncased'
    for root, dirs, files in os.walk(local_path):
        for file in files:
            s3_file_path = os.path.join(root.replace(local_path, s3_path), file)
            local_file_path = os.path.join(root, file)
            s3.upload_file(local_file_path, bucket_name, s3_file_path)
            print(s3_file_path)

if __name__ =='__main__':
    upload_dataset()
    print('upload finished! ')
    upload_bert()
    print('upload bert finished! ')