from argparse import ArgumentParser

def argparser():
    ap = ArgumentParser()
    
    ap.add_argument("--train_path", type=str)
    ap.add_argument("--val_path", type=str)
    ap.add_argument("--model_save_path", type=str)
    
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr_warmup", type=float,default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--max_grad_norm", type=float, default=1.00)
    ap.add_argument("ne_count", type=int, default=100)
    ap.add_argument("nr_count", type=int, default=100)
    
    ap.add_argument("--config_path", type=str)
    ap.add_argument("--tokenizer_path", type=str)
    ap.add_argument("--freeze_model_layers", type=bool, default=True)
    ap.add_argument("--sample_size", type=int, default=50)
    ap.add_argument("--convert_to_lowercase", type=bool, default=True)
    ap.add_argument("--max_span_size", type=int, default=15)
    
    ap.add_argument("--debug", type=bool, default=False) # Need to integrate logging, does nothing for now
    
    return ap
    