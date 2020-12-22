import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--steps', default=5005, type=int)
parser.add_argument('--time', default=0.545, type=float)
parser.add_argument('--epochs', default=90, type=int)
args = parser.parse_args()

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )

def display_time(seconds):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result)

print(display_time(args.time * args.steps * args.epochs))
