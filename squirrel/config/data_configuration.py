from squirrel.config import register_config


@register_config('data')
def build_data_config(parser):
    parser.add_argument('--logfile', type=str, default=None)

    parser.add_argument(
        '--dataloader',
        type=str,
        default=None,
        choices=['default', 'multi', 'order', 'target_noise'])

    # data-field
    parser.add_argument(
        '--source_field',
        default='text',
        type=str,
        choices=['text', 'image_feature'],
        help='field of source information')

    # data_path
    parser.add_argument(
        '--data_prefix', type=str, default='/data0/data/transformer_data/')
    parser.add_argument('--workspace_prefix', type=str, default='./')
    parser.add_argument(
        '--dataset', type=str, default='iwslt', help='"flickr" or "iwslt"')

    parser.add_argument(
        '--src', type=str, default=None, help='source language marker')
    parser.add_argument(
        '--trg', type=str, default=None, help='target language marker')
    parser.add_argument(
        '--test_src',
        type=str,
        default=None,
        help='source language marker in testing')
    parser.add_argument(
        '--test_trg',
        type=str,
        default=None,
        help='target language marker in testing')
    parser.add_argument(
        '--track_best', type=str, default=None, help='efault track all.')

    # files
    parser.add_argument(
        '--train_set', type=str, default=None, help='which train set to use')
    parser.add_argument(
        '--dev_set', type=str, default=None, help='which dev set to use')
    parser.add_argument(
        '--test_set', type=str, default=None, help='which test set to use')
    parser.add_argument(
        '--suffixes',
        type=str,
        nargs='*',
        default=['.src', '.trg'],
        help='all the files should have the same suffix')

    # vocabulary
    parser.add_argument(
        '--vocab_file', type=str, default=None, help='user-defined vocabulary')
    parser.add_argument(
        '--max_vocab_size',
        type=int,
        default=50000,
        help='max vocabulary size')

    # initial tokens
    parser.add_argument(
        '--lang_as_init_token',
        action='store_true',
        help='use language token as initial tokens')
    parser.add_argument(
        '--force_translate_to',
        type=str,
        default=None,
        help='force my decode to decode to X langauge.')
    parser.add_argument(
        '--force_translate_from',
        type=str,
        default=None,
        help='force my decode to decode from X langauge.')
    parser.add_argument(
        '--lm_only',
        action='store_true',
        help='if true, the model will ignore the source.')

    # details
    parser.add_argument(
        '--sample_a_training_set',
        action='store_true',
        help='if we have multiple training sets.')

    parser.add_argument(
        '--remove_init_eos',
        action='store_true',
        help='possibly remove <init>/<eos> tokens for encoder-decoder')

    parser.add_argument(
        '--load_lazy',
        action='store_true',
        help='load a lazy-mode dataset, not save everything in the mem')

    return parser


@register_config('data_noise')
def build_noisy_dataloader_config(parser):

    parser.add_argument(
        '--noise_dataflow', type=str, default='trg', choices=['src', 'trg'])
    parser.add_argument(
        '--noise_types',
        nargs='*',
        default=None,
        choices=[
            'word_shuffle', 'word_dropout', 'word_blank',
            'word_dropout_at_anywhere'
        ],
        help='use noise generator in the dataloader')

    parser.add_argument('--noise_dropout_prob', type=float, default=0.0)
    parser.add_argument('--noise_shuffle_distance', type=float, default=3)
    parser.add_argument('--noise_blank_prob', type=float, default=0.0)
    parser.add_argument('--noise_blank_word', type=str, default='<unk>')

    parser.add_argument(
        '--output_suggested_edits',
        action='store_true',
        help='output suggested edits')

    return parser
