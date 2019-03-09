from squirrel.config import register_config


@register_config('utils')
def build_utils_config(parser):
    parser.add_argument(
        '--debug',
        action='store_true',
        help='debug mode: no saving or tensorboard')
    parser.add_argument(
        '--no_valid',
        action='store_true',
        help='debug mode: no validation during training')
    parser.add_argument(
        '--valid_ppl',
        action='store_true',
        help='debug mode: validation with ppl')
    '''
    Add some distributed options. For explanation of dist-url and dist-backend please see
    http://pytorch.org/tutorials/intermediate/dist_tuto.html
    --local_rank will be supplied by the Pytorch launcher wrapper (torch.distributed.launch)
    '''
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--distributed", default=False, type=bool)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--init_method", default=None, type=str)
    parser.add_argument("--master_port", default=11111, type=int)

    # load pre_saved arguments
    parser.add_argument("--json", default=None, type=str)

    return parser
