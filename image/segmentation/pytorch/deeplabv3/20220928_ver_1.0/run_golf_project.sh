python main.py \
    /home/kimyh/python/ai/image/segmentation/project/golf \
    phase_1st \
    vworld_get_full_class \
    --network=deeplabv3_resnet101 \
    background,F,G,TUP,SB,WH,CART \
    --batch_size=8 \
    --device_num=0 \
    --epochs=100 \
    --optimizer_name=AdamW \
    --loss_name=CrossEntropyLoss \
    --scheduler_name=ExponentialLR \
    --learning_rate=2e-3 \
    --gamma=0.98 \
    --trained_weight=test_phase/vworld_test_model/0000/epoch0009/weight.pt \
    --start_epoch=43 \