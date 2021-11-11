export S3DIS_DIR=/home/andrej/datasets/Stanford3dDataset_v1.2
export OUTPUT_DIR=/home/andrej/datasets/S3DIS_preprocessed
export HYDRA_FULL_ERROR=1

preprocess() {
    filename=$1
    python src/preprocess.py \
        dataset_dir=$S3DIS_DIR \
        output_dir=$OUTPUT_DIR \
        dataset=s3dis \
        sens_file=$filename
}

export -f preprocess

# preprocess
parallel -j 16 --linebuffer time preprocess ::: `find $S3DIS_DIR -mindepth 2 -maxdepth 2 -type d`