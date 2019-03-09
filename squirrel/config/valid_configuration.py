from squirrel.config import register_config


@register_config('valid')
def build_valid_config(parser):
    parser.add_argument(
        '--valid_batch_size',
        type=int,
        default=2048,
        help='# of tokens processed per batch')

    parser.add_argument(
        '--valid_maxlen',
        type=int,
        default=10000,
        help='limit the train set sentences to this many tokens')

    parser.add_argument(
        '--length_ratio',
        type=int,
        default=3,
        help='maximum lengths of decoding')

    parser.add_argument(
        '--beam_size',
        type=int,
        default=1,
        help='beam-size used in Beamsearch, default using greedy decoding')

    parser.add_argument(
        '--alpha', type=float, default=1, help='length normalization weights')
    parser.add_argument(
        '--original',
        action='store_true',
        help='output the original output files, not the tokenized ones.')
    parser.add_argument(
        '--decode_test',
        action='store_true',
        help='evaluate scores on test set instead of using dev set.')
    parser.add_argument(
        '--output_decoding_files',
        action='store_true',
        help='output separate files in testing.')
    parser.add_argument(
        '--output_on_the_fly',
        action='store_true',
        help='decoding output and on the fly output to the files')
    parser.add_argument(
        '--decoding_path',
        type=str,
        default=None,
        help='manually provide the decoding path for the models to decode')

    parser.add_argument(
        '--output_confounding',
        action='store_true',
        help='language confounding.')

    parser.add_argument(
        '--metrics',
        nargs='*',
        type=str,
        help='metrics used in this task',
        default=['BLEU'],
        choices=[
            'BLEU', 'GLEU', 'RIBES', 'CODE_MATCH', 'TER', 'METEOR', 'CIDER'
        ])

    parser.add_argument(
        '--real_time', action='store_true', help='real-time translation.')

    return parser
