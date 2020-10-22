#cars196
python train.py -b 32 --gpu 2 --dataset cars196 --freeze_BN --epochs 50 --dim 512 --r 0.05 -C 98 --N 12 --weight_lambda 0.3 --centerlr 0.03 --rate 0.1 --new_epoch_to_decay 20 40

#cub200
python train.py -b 32 --gpu 2 --dataset cub200 --freeze_BN --epochs 50 --dim 512 --r 0.05 -C 100 --N 12 --weight_lambda 0.3 --centerlr 0.03 --rate 0.1 --new_epoch_to_decay 20 40


#sop
python train.py -b 32 --gpu 2 --dataset online_products --epochs 50 --dim 512 --r 0.05 -C 11318 --N 1 --weight_lambda 0.0 --centerlr 0.3 --rate 0.1 --new_epoch_to_decay 20 40