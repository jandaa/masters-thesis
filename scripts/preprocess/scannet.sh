export SCANNET_DIR=/media/starslab/datasets/scannet
export OUTPUT_DIR=/media/starslab/datasets/scannet_preprocessed_2cm_fpfh_cluster_no_images
export HYDRA_FULL_ERROR=1

preprocess() {
    filename=$1
    python src/preprocess.py \
        hydra.run.dir=outputs/preprocess \
        dataset_dir=$SCANNET_DIR \
        output_dir=$OUTPUT_DIR \
        dataset=scannet \
        dataset.voxel_size=0.02 \
        dataset.name=scannetv2_pretrain \
        sens_file=$filename
}

export -f preprocess

# preprocess
parallel -j 4 --linebuffer time preprocess ::: `find $SCANNET_DIR/scans/ -mindepth 1 -maxdepth 1 -type d`