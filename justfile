train:
    uv run python src/denoiser/train.py \
        --train_data_path "data/CC15" \
        --clean_img_keyword "_mean" \
        --noisy_img_keyword "_real" \
        --model_name "vit_base_patch16_224" \
        --batch_size 4 \
        --cropsize 512 \
        --learning_rate 0.0001 \
        --iteration 1000 \
        --interval 100 \
        --output_dir "./results" \
        --log_dir "logs" \

train_minimal:
    uv run python src/denoiser/train.py \
        --train-data-path "data/CC15" \
        --clean-img-keyword "_mean" \
        --noisy-img-keyword "_real" \
        --batch-size 4 \
        --cropsize 128 \
        --learning-rate 0.0001 \
        --iteration 10 \
        --interval 5 \
        --output-dir "./results_minimal" \
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
