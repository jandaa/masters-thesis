import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    semantics()
