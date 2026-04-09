train:
    train \
        --train-data-path "data/CC15" \
        --clean-img-keyword "_mean" \
        --noisy-img-keyword "_real" \
        --model-name "vit_unet" \
        --batch-size 4 \
        --crop-size 256 \
        --learning-rate 0.001 \
        --iteration 5000 \
        --interval 100 \
        --output-dir "./results" \
        --log-dir "logs" \

train_minimal:
    train \
        --train-data-path "data/CC15" \
        --clean-img-keyword "_mean" \
        --noisy-img-keyword "_real" \
        --model-name "resnet34" \
        --loss-type "mse" \
        --batch-size 4 \
        --crop-size 256 \
        --learning-rate 0.0001 \
        --iteration 5000 \
        --interval 100 \
        --output-dir "./results_mse" \
        --log-dir "logs" \

train_base:
    train \
        --train-data-path "/home/s.chochi/ai-works/datasets/data/DIV2K" \
        --model-name "vit_unet" \
        --loss-type "ier" \
        --noise-sigma 25 \
        --batch-size 4 \
        --crop-size 256 \
        --learning-rate 0.0001 \
        --iteration 100 \
        --interval 5 \
        --output-dir "./results_vit_div2k" \
        --log-dir "logs" \

inference:
    uv run python src/denoiser/inference.py \
        --model "./results/best_model.pth" \
        --input "./data/CC15_inf" \
        --output "./results/inf"

data_split:
    uv run python src/denoiser/make_train_val_split.py \
        --data-dir "data/CC15" \
        --clean-img-keyword "_mean" \
        --noisy-img-keyword "_real" \
