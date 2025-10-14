#  python src/denoiser/train.py \
#  --train_data_path "/home/s.chochi/noise-translator/data/CC15" \
#  --clean_img_keyword "_mean" \
#  --noisy_img_keyword "_real" \
#  --batch_size 4 \
#  --cropsize 128 \
#  --learning_rate 0.0001 \
#  --iteration 30 \
#  --interval 5 \
#  --output_dir "./results" \
#  --log_dir "logs" \
#  --tensorboard True \
#  --verbose "info"

# For inference
python src/denoiser/inference.py \
 --model "./results/best_model.pth" \
 --input "./data/CC15_inf" \
 --output "./results/inf"

