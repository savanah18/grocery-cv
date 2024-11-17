from ultralytics import YOLO
import hydra

# success
# aiml/outputs/2024-11-11/08-24-34
# aiml/outputs/2024-11-11/19-38-29

# using hydra create a train function 
@hydra.main(config_path="config", config_name="train_config_006")
def train(cfg):
    print(cfg)
    model = YOLO(model="yolo11n.pt")
    model.train(**cfg.train)

if __name__ == "__main__":
    train()