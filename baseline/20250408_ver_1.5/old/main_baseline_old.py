'''
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pandas
pip install tqdm
pip install -U scikit-learn
'''
import sys
import os
sys.path.append(os.getcwd())
import argparse
import json
import warnings
warnings.filterwarnings('ignore')



from models import network as net
from mylocalmodules import dataloader as dam

sys.path.append('/home/kimyh/python/ai')
from sharemodule import lossfunction as lfm
from sharemodule import train as trm
from sharemodule import trainutils as tum
from sharemodule import logutils as lom
from sharemodule import utils as utm


def main():
    # pre-requisite
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path')
    
    parser.add_argument('--phase')
    parser.add_argument('--dataset_name')
    parser.add_argument('--purpose', type=str)

    # train variable
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--loss_function_name', type=str, default='CrossEntropyLoss')
    parser.add_argument('--optimizer_name', type=str, default='SGD')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='optimizer learning rate')
    parser.add_argument('--scheduler_name', type=str)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.95)

    args = parser.parse_args()
    args_setting = vars(args)
    utm.envs_setting(42)
    
    
    root_path = args.root_path
    phase = args.phase
    dataset_name = args.dataset_name
    purpose = args.purpose
    
    device_num = args.device_num
    epochs = args.epochs
    batch_size = args.batch_size
    max_grad_norm = args.max_grad_norm 
    dropout_p = args.dropout_p
    loss_function_name = args.loss_function_name
    optimizer_name = args.optimizer_name
    learning_rate = args.learning_rate
    scheduler_name = args.scheduler_name
    gamma = args.gamma
    random_seed = args.random_seed
    
    shuffle = args.shuffle
    drop_last = args.drop_last
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    
    # =========================================================================
    # log 생성
    log = lom.get_logger(
        get='TRAIN',
        root_path=root_path,
        log_file_name=f'train.log',
        time_handler=True
    )
    whole_start = time.time()
    
    # =========================================================================
    print('get dataloader')
    Train_Dataloader = dam.get_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    Val_Dataloader = dam.get_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    # ===================================================================    
    print('get_model')
    device = tum.get_device(device_num)
    model = net()
    model.to(device)
    
    # ===================================================================    
    optimizer = tum.get_optimizer(
        base='torch',
        method=optimizer_name,
        model=model,
        learning_rate=learning_rate
    )
    # loss function
    loss_function = lfm.LossFunction(
        base='torch',
        method='MSE'
    )
    scheduler = tum.get_scheduler(
        base='torch',
        method=scheduler_name,
        optimizer=optimizer,
        gamma=gamma
    )
    # ============================================================
    # save setting
    root_save_path = f'{root_path}/trained_model/{phase}/{dataset_name}'
    condition_order = tum.get_condition_order(
        args_setting=args_setting,
        save_path=root_save_path,
        except_arg_list=['epochs']
    )
    model_save_path = f'{root_save_path}/{condition_order}'
    os.makedirs(f'{root_save_path}/{condition_order}', exist_ok=True)
    with open(f'{root_save_path}/{condition_order}/args_setting.json', 'w', encoding='utf-8') as f:
        json.dump(args_setting, f, indent='\t', ensure_ascii=False)
    
    # ============================================================
    # train
    print('train...')    
    print(f'condition order: {condition_order}')
    _ = trm.train(
        model=model, 
        purpose=purpose, 
        start_epoch=0, 
        epochs=epochs, 
        train_dataloader=Train_Dataloader, 
        validation_dataloader=Val_Dataloader, 
        uni_class_list=None, 
        device=device, 
        loss_function=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        max_grad_norm=max_grad_norm, 
        reset_class=None, 
        model_save_path=model_save_path
    )

        
if __name__  == '__main__':
    main()