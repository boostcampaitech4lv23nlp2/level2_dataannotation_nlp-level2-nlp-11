import argparse
import sys
from typing import Tuple

import inference
import train
from omegaconf import OmegaConf, dictconfig

if __name__ == "__main__":
    """
    argparse 파라미터 설명
        --mode -m : 실행 모드를 지정
                    train : [train, t] 모델 학습
                    inference : [inference, i ] 모델 예측
                    exp : [exp , e]  모델 실험(sweep) / 구현예정
        --config -c : config.yaml 의 file_name
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", required=True)
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    # pip install omegaconf 부터
    conf = OmegaConf.load(f"../config/{args.config}.yaml")

    if args.mode == "train" or args.mode == "t":
        # conf.k_fold 변수 확인
        if conf.k_fold.use_k_fold:
            train.k_train(conf)
        else:
            train.train(conf)

    elif args.mode == "inference" or args.mode == "i":
        inference.main(conf)
    else:
        print("모드를 다시 설정해주세요 ")
        print("train     : t,\ttrain")
        print("inference : i,\tinference")
