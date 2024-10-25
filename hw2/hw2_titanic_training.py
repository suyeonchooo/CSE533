# Kaggle Competition
# Titanic - Machine Learning from Disaster

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse

# 기존 Titanic 데이터 로드를 위한 모듈을 임포트
from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print(BASE_PATH, "!!!!!!!")

import sys
sys.path.append(BASE_PATH)

# Titanic dataset 관련 함수 임포트 (수정된 부분)
from titanic_dataset import TitanicDataset, TitanicTestDataset, get_preprocessed_dataset

def get_data():
    # 전처리된 Titanic 데이터셋을 가져옴 (수정된 부분)
    train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()

    print(len(train_dataset), len(validation_dataset), len(test_dataset))

    # wandb 설정에 따라 데이터 로더 생성 (배치 크기 및 셔플 여부 설정)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

    return train_data_loader, validation_data_loader

    
class MyModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_unit_list=[20, 20]):
        super().__init__()
        # wandb 대신 n_hidden_unit_list를 직접 인자로 받아서 사용
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden_unit_list[0]),
            nn.PReLU(),
            nn.Linear(n_hidden_unit_list[0], n_hidden_unit_list[1]),
            nn.PReLU(),
            nn.Linear(n_hidden_unit_list[1], n_output),
        )

    def forward(self, x):
        return self.model(x)


def get_model_and_optimizer():
    # Titanic 데이터에 맞는 입력 (피처 수 11)과 출력 (생존 여부 2 클래스)을 설정 (수정된 부분)
    my_model = MyModel(n_input=11, n_output=2)
    optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

    return my_model, optimizer


from pathlib import Path

def training_loop(model, optimizer, train_data_loader, validation_data_loader, save_interval=100):
    n_epochs = wandb.config.epochs
    loss_fn = nn.CrossEntropyLoss()  
    next_print_epoch = 100

    # 모델 파일을 저장할 "epoch/activation_function_name" 폴더 생성
    save_dir = Path("epoch") / "ReLU"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        num_trains = 0
        for train_batch in train_data_loader:
            input = train_batch['input']
            target = train_batch['target']
            output_train = model(input)
            loss = loss_fn(output_train, target)
            loss_train += loss.item()
            num_trains += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_validation = 0.0
        num_validations = 0
        with torch.no_grad():
            for validation_batch in validation_data_loader:
                input = validation_batch['input']
                target = validation_batch['target']
                output_validation = model(input)
                loss = loss_fn(output_validation, target)
                loss_validation += loss.item()
                num_validations += 1

        # wandb에 로그를 기록
        wandb.log({
            "Epoch": epoch,
            "Training loss": loss_train / num_trains,
            "Validation loss": loss_validation / num_validations
        })

        # 에포크마다 출력 (100 에포크마다 출력)
        if epoch >= next_print_epoch:
            print(
                f"Epoch {epoch}, "
                f"Training loss {loss_train / num_trains:.4f}, "
                f"Validation loss {loss_validation / num_validations:.4f}"
            )
            next_print_epoch += 100

        # 모델 저장 (save_interval마다 저장)
        if epoch % save_interval == 0 or epoch == n_epochs:
            model_save_path = save_dir / f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")


def main(args):
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [20, 20],
    }

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="DL_hw2_titanic_train",
        notes="Titanic dataset experiment",  # 프로젝트 관련 내용 업데이트 (수정된 부분)
        tags=["my_model", "titanic"],  # 태그 업데이트 (수정된 부분)
        name=current_time_str,
        config=config
    )
    print(args)
    print(wandb.config)

    # Titanic 데이터를 사용해 훈련 및 검증 데이터 로더 생성 (수정된 부분)
    train_data_loader, validation_data_loader = get_data()

    # 모델과 옵티마이저 설정
    linear_model, optimizer = get_model_and_optimizer()

    print("#" * 50, 1)

    # 모델 훈련 루프 실행
    training_loop(
        model=linear_model,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader
    )
    wandb.finish()


# argparse를 사용해 커맨드 라인에서 인자를 받을 수 있도록 설정
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=512, help="Batch size (int, default: 512)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=1_000, help="Number of training epochs (int, default:1_000)"
    )

    args = parser.parse_args()

    # 메인 함수 호출
    main(args)
