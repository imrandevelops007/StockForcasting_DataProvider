from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.stock_kaggle import Dataset_StockKaggle

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    #'StockKaggle': Dataset_StockKaggle
}  

def data_provider(args, flag):
    if args.data == 'StockKaggle':  # ✅ Handle custom dataset early
        size = (args.seq_len, args.label_len, args.pred_len)
        timeenc = 0 if args.embed != 'timeF' else 1

        data_set = Dataset_StockKaggle(
            root_path=args.root_path,
            flag=flag,
            size=size,
            file_name=args.file_name,  # ✅ Needed from CLI
            features=args.features,
            target=args.target,
            scale=True,
            timeenc=timeenc,
            freq=args.freq,
            train_ratio=0.7,
            val_ratio=0.15
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=(flag == 'train'),
            num_workers=args.num_workers
        )

        return data_set, data_loader

    # ✅ This block is skipped if using StockKaggle
    Data = data_dict[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader

    else:
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=True,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader



""" def data_provider(args, flag):
    Data = data_dict[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'StockKaggle':
         size = (args.seq_len, args.label_len, args.pred_len)
    timeenc = 0 if args.embed != 'timeF' else 1

    data_set = Dataset_StockKaggle(
        root_path=args.root_path,
        flag=flag,
        size=size,
        file_name=args.file_name,         # ✅ from CLI
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        train_ratio=0.7,
        val_ratio=0.15
    )

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=(flag == 'train'),
        num_workers=args.num_workers
    )

    return data_set, data_loader """
            
""" if args.data == 'm4':
            drop_last = False
            
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader """
