import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run JME.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='toy', help='Choose a dataset from {toy}')
    parser.add_argument('--behavior_data', nargs='?',default='["train_view.txt","train_fav.txt","train.txt"]',
        help='Behavior data, the target behavior should be last.')
    parser.add_argument('--kge', nargs='?', default='trans_e', help='Choose a KGE method from {trans_e,trans_h,trans_r,dist_mult,compl_ex,kg2e,conv_e}')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--dim', type=int, default=64, help='Embedding size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Number of epoch for early stopping.')
    parser.add_argument('--Ks', nargs='?', default='[5,10,20]', help='Calculate metric@K when evaluating.')
    parser.add_argument('--model_path', nargs='?', default='', help='Model path for evaluation.')
    parser.add_argument('--use_boac', type=int, default=1, help='0: Without Behavior Overlap Aware Converter, 1: Full model.')
    parser.add_argument('--use_bam', type=int, default=1, help='0: Without Behavior Aware Margin Function, 1: Full model.')
    parser.add_argument('--use_epl', type=int, default=1, help='0: Without EPL module, 1: Full model.')
    parser.add_argument('--neg_size', type=int, default=1, help='Negative sampling size.')

    args = parser.parse_args()

    save_dir = f'{args.dataset}_dim{args.dim}_lr{args.lr}_{args.kge}'
    save_path = f'trained_model/{save_dir}/'
    args.save_path = save_path

    return args
