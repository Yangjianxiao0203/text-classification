sudo s3fs deeplearningbucket ./s3 -o passwd_file=$(pwd)/bucket_pass -o allow_other

sudo usermod -a -G root ubuntu