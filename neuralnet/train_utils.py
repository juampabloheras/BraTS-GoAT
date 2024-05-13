import argparse

def load_fold_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def none_or_int(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer or 'None'")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=none_or_int, default=None, help="An optional integer argument or 'None'")
    parser.add_argument('--train_dir', type=str, nargs='+', required=True)
    parser.add_argument('--test_dir', type=str, nargs='+', required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--loss', type=str, nargs='+', required=True)
    parser.add_argument('--weights', type=float, nargs='+', required=True)
    parser.add_argument('--loss_weights', type=float, nargs='+', required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--partial_file_names_dir', type=str, nargs='+', required=True)
    parser.add_argument('--folds_dir', type=str)
    parser.add_argument('--fold_no', type=int)
    parser.add_argument('--cluster_dict', type=str)


    # Optional arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=6e-05)
    parser.add_argument('--power', type=float, default=0.75)

    eval_parser = parser.add_mutually_exclusive_group(required=False)
    eval_parser.add_argument('--eval_on_overlap', dest='eval_on_overlap', action='store_true')
    eval_parser.add_argument('--no-eval_on_overlap', dest='eval_on_overlap', action='store_false') 
    parser.set_defaults(eval_on_overlap=True) # Default is True

    # Choose if train on overlapping or disjoint regions
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--train_overlap', dest='train_on_overlap', action='store_true')
    feature_parser.add_argument('--no-train_overlap', dest='train_on_overlap', action='store_false')
    parser.set_defaults(train_on_overlap=True) # Default is True

    # Parse the arguments
    args = parser.parse_args()

    # Get variables
    alpha = args.alpha
    train_dir = args.train_dir
    test_dir = args.test_dir
    ckpt_dir = args.ckpt_dir
    out_dir = args.out_dir
    loss_str = args.loss
    weights = args.weights
    loss_weights = args.loss_weights
    model_str = args.model
    partial_file_names = args.partial_file_names_dir
    folds_dir = args.folds_dir
    fold_no = args.fold_no
    cluster_dict = args.cluster_dict

    max_epoch = args.epochs
    lr = args.lr
    power = args.power

    eval_on_overlap = args.eval_on_overlap
    train_on_overlap = args.train_on_overlap

    return (alpha, train_dir, test_dir, ckpt_dir, out_dir, loss_str, weights, loss_weights, 
            model_str, partial_file_names, folds_dir, fold_no, cluster_dict, max_epoch, lr, power, eval_on_overlap, train_on_overlap)