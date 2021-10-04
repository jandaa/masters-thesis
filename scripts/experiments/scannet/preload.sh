export SCANNET_DIR=/media/starslab/datasets/scannet
export HYDRA_FULL_ERROR=1

preprocess() {
    filename=$1
    python src/preprocess.py \
        dataset_dir=$SCANNET_DIR \
        dataset=scannet \
        sens_file=$filename \
        force_reload=True
}

export -f preprocess

# preprocess
parallel -j 8 --linebuffer time preprocess ::: `find $SCANNET_DIR/scans/scene*/*.sens`