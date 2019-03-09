from squirrel.config import register_config


@register_config('train')
def build_train_config(parser):
    # basic settings
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument(
        '--disable_lr_schedule',
        action='store_true',
        help='disable the transformer-style learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=25)

    parser.add_argument(
        '--sub_inter_size',
        type=int,
        default=1,
        help='process multiple batches before one update')
    parser.add_argument(
        '--inter_size',
        type=int,
        default=4,
        help='process multiple batches before one update')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='# of tokens processed per batch')

    parser.add_argument(
        '--label_smooth',
        type=float,
        default=0.1,
        help='regularization via label-smoothing during training.')
    parser.add_argument(
        '--eval_every',
        type=int,
        default=1000,
        help='run evaluation (validation) for every several iterations')
    parser.add_argument(
        '--print_every',
        type=int,
        default=0,
        help='print every during training')
    parser.add_argument(
        '--att_plot_every',
        type=int,
        default=0,
        help='visualization the attention matrix of a sampled training set.')
    parser.add_argument(
        '--save_every',
        type=int,
        default=50000,
        help='save the best checkpoint every 50k updates')
    parser.add_argument(
        '--maximum_steps',
        type=int,
        default=300000,
        help='maximum steps you take to train a model')

    parser.add_argument(
        '--lm_steps',
        type=int,
        default=0,
        help='pre-training steps without encoder inputs.')
    parser.add_argument('--lm_schedule', action='store_true')
    parser.add_argument(
        '--maxlen',
        type=int,
        default=10000,
        help='limit the train set sentences to this many tokens')

    parser.add_argument('--maxatt_size', type=int, default=2200000)

    return parser
