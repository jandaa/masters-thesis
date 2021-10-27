export SCANNET_DIR=/home/ajanda/scratch/datasets/scannet
export OUTPUT_DIR=/home/ajanda/scratch/datasets/scannet_preprocessed_2mm
export HYDRA_FULL_ERROR=1

preprocess() {
    filename=$1
    python src/preprocess.py \
        dataset_dir=$SCANNET_DIR \
        output_dir=$OUTPUT_DIR \
        dataset=scannet \
        dataset.pretrain.voxel_size=0.02 \
        sens_file=$filename \
        force_reload=True
}

export -f preprocess

# preprocess
parallel -j 20 --linebuffer time preprocess ::: `find $SCANNET_DIR/scans/scene*/*.sens`