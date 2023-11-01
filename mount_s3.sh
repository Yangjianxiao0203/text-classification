sudo apt install awscli
sudo apt install s3fs

chmod 600 bucket_pass

mkdir s3

sudo s3fs deeplearningbucket ./s3 -o passwd_file=$(pwd)/bucket_pass -o allow_other

sudo usermod -a -G root ubuntu