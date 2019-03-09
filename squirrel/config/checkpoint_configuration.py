from squirrel.config import register_config


@register_config('checkpoint')
def build_checkpoint_config(parser):
    parser.add_argument(
        '--load_from', type=str, default='none', help='load from checkpoint')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='when loading from the saved model, it resumes from that.')

    parser.add_argument(
        '--sweep_checkpoints',
        action='store_true',
        help='looping.. load the checkpoints (only for test)')
    parser.add_argument(
        '--never_load',
        nargs='*',
        type=str,
        default=None,
        help='parameters contaiend in this list will be discarded.')
    parser.add_argument(
        '--freeze',
        nargs='*',
        type=str,
        help='freeze part of the loaded parameters from the checkpoints.')

    parser.add_argument(
        '--tensorboard', action='store_true', help='use TensorBoard')

    return parser
