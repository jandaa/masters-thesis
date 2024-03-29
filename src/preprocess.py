import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig
from datasets.interface import DataInterfaceFactory

log = logging.getLogger("preprocess")


@hydra.main(config_path="config", config_name="preprocess")
def preprocess(cfg: DictConfig) -> None:

    # Extract scene name
    scene_path = Path(cfg.sens_file)
    scene_name = scene_path.name

    # Make datapoint out of scene
    log.info("Loading data point")
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()
    datapoint = data_interface.get_datapoint(scene_path)

    log.info(f"Preprocessing scene: {scene_name}")
    data_interface.preprocess(datapoint)

    log.info(f"Finished preprocessing scene: {scene_name}")


if __name__ == "__main__":
    preprocess()
