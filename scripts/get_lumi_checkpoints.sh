base_path=$1
dataset_name=$2
output_dir=$3

for subdir in $(aws s3 ls $base_path | awk '{print $2}' | grep -- "$dataset_name"); do
    for checkpoint in $(aws s3 ls $base_path$subdir | awk '{print $2}' | grep unsharded | grep -v latest); do
        echo $base_path$subdir$checkpoint
        aws s3 sync $base_path$subdir$checkpoint $output_dir$checkpoint
    done
done