export SCANNET_DIR=/media/starslab/datasets/scannet
export OUTPUT_DIR=/media/starslab/datasets/scannet_preprocessed_2cm_fpfh
export HYDRA_FULL_ERROR=1

preprocess() {
    filename=$1
    python src/preprocess.py \
        dataset_dir=$SCANNET_DIR \
        output_dir=$OUTPUT_DIR \
        dataset=scannet \
        dataset.voxel_size=0.02 \
        sens_file=$filename
}

export -f preprocess

# preprocess
parallel -j 4 --linebuffer time preprocess ::: `find $SCANNET_DIR/scans/ -mindepth 1 -maxdepth 1 -type d`