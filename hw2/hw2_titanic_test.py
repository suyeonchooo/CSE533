import torch
import pandas as pd
import argparse
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from titanic_dataset import TitanicDataset, TitanicTestDataset, get_preprocessed_dataset
from hw2_titanic_training import MyModel

def get_test_data_loader(batch_size):
    # 전처리된 데이터셋에서 테스트 데이터 준비
    _, _, test_dataset = get_preprocessed_dataset()  # get_preprocessed_dataset에서 테스트 데이터셋 반환
    
    # 테스트 데이터를 위한 DataLoader 생성
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_data_loader

def load_model_for_testing(model_file):
    # wandb.config에서 n_hidden_unit_list를 가져와서 모델 구성
    model = MyModel(n_input=11, n_output=2, n_hidden_unit_list=wandb.config.n_hidden_unit_list)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

def test_and_generate_submission(model, test_data_loader, submission_file="submission.csv"):
    predictions = []
    
    # 모델을 사용하여 테스트 데이터 예측
    with torch.no_grad():
        for batch in test_data_loader:
            input_data = batch['input']
            output = model(input_data)
            predicted_class = output.argmax(dim=1)
            predictions.extend(predicted_class.cpu().numpy())
    
    # test.csv 파일에서 PassengerId를 직접 로드
    test_data_path = Path(__file__).parent / "test.csv"
    test_data = pd.read_csv(test_data_path)
    
    # 예측 결과를 DataFrame으로 생성하여 저장
    output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
    output.to_csv(submission_file, index=False)
    print("Your submission was successfully saved!")

def main(args):
    # wandb 초기화 (테스트용 모드로 설정)
    config = {
        'epochs': 1000,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [20, 20],
    }

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="DL_hw2_titanic_test",
        config=config
    )

    # 모델 파일 경로 설정 (epoch/PReLU 폴더에서 특정 epoch의 모델 파일 불러오기)
    model_file = Path("epoch") / "PReLU" / f"model_epoch_{args.epoch}.pth"
    loaded_model = load_model_for_testing(model_file)

    # 테스트 데이터 로더 생성
    test_data_loader = get_test_data_loader(batch_size=wandb.config.batch_size)

    # submission 파일 경로 설정
    submission_file = args.submission_file if args.submission_file else Path("/data/2_data_server/nlp-05/KOREATECH/2024-2_DL/link_dl/_03_your_code/CSE533/hw2/submission.csv")

    # 저장된 모델로 예측하고 submission 파일 생성
    test_and_generate_submission(model=loaded_model, test_data_loader=test_data_loader, submission_file=submission_file)
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Model Testing and Submission Generator")
    
    # 필요한 인자 추가
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False")
    parser.add_argument("--submission_file", type=str, help="Path to save the submission file")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the model to load")

    args = parser.parse_args()
    
    # 메인 함수 실행
    main(args)

