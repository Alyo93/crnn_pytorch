python train.py --dataset_name synth-chinese --batch_size 512 --lr 0.001 --optimizer adam \
--train_dir /home/hxt/Synth-Chinese/Sythetic_String_Dataset  \
--val_dir /home/hxt/Synth-Chinese/Sythetic_String_Dataset  \
--chars_file /home/hxt/projects/crnn_my/chars/char_std_5990.txt \
--gpus 1,2,3 --num_workers 4 \

