from squirrel.config import register_config


@register_config('model_basic_transformer')
def build_model_baisc_transformer_config(parser):
    parser.add_argument(
        '--model', type=str, default='Transformer')  # current stage
    parser.add_argument(
        '--base',
        type=str,
        default='bpe',
        choices=['byte', 'char', 'bpe', 'wp'])
    parser.add_argument(
        '--prefix',
        type=str,
        default='[time]',
        help='prefix to denote the model, nothing or [time]')
    parser.add_argument(
        '--params',
        type=str,
        default='customize',
        help='pamarater sets: james-iwslt, t2t-base')

    # customize
    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='basic parameter of the model size')
    parser.add_argument(
        '--d_hidden',
        type=int,
        default=2048,
        help='used in feedforward network')
    parser.add_argument(
        '--d_embedding',
        type=int,
        default=512,
        help='embedding size for input/output')
    parser.add_argument(
        '--warmup',
        type=int,
        default=4000,
        help='warming-up steps during training')
    parser.add_argument(
        '--n_layers',
        type=int,
        default=6,
        help='number of encoder-decoder (assume the same)')
    parser.add_argument(
        '--n_heads',
        type=int,
        default=8,
        help='number of heads for multi-head attention')
    parser.add_argument(
        '--n_cross_heads',
        type=int,
        default=8,
        help='number of heads for multi-head attention')
    parser.add_argument(
        '--drop_ratio',
        type=float,
        default=0.1,
        help='dropout ratio for everything else')
    parser.add_argument(
        '--input_drop_ratio',
        type=float,
        default=0.1,
        help='dropout ratio for input embeddings')
    parser.add_argument(
        '--relu_drop_ratio',
        type=float,
        default=0.0,
        help='dropout ratio used in feed-forward networks')
    parser.add_argument(
        '--attn_drop_ratio',
        type=float,
        default=0.0,
        help='dropout ratio used in all attention weights')

    return parser


@register_config('model_ablation_transformer')
def build_model_ablation_transformer_config(parser):
    parser.add_argument(
        '--block_order',
        type=str,
        default='tdan',
        choices=['tdan', 'tdna', 'tnda'])
    parser.add_argument(
        '--normalize_emb',
        action='store_true',
        help='normalize embedding (IO)')
    parser.add_argument(
        '--causal_enc',
        action='store_true',
        help='use unidirectional encoder (useful for real-time translation)')
    parser.add_argument(
        '--encoder_lm',
        action='store_true',
        help='use unidirectional encoder with additional loss as a LM')
    parser.add_argument(
        '--causal', action='store_true', help='use causal attention')
    parser.add_argument(
        '--cross_attn_fashion',
        type=str,
        default='forward',
        choices=['forward', 'reverse', 'last_layer'])
    parser.add_argument(
        '--share_embeddings',
        action='store_true',
        help='share embeddings between encoder and decoder')
    parser.add_argument(
        '--share_encdec',
        action='store_true',
        help='completely share the encoder with the decoder.')
    parser.add_argument(
        '--uniform_embedding_init',
        action='store_true',
        help='by default, we use Transformer clever init for embeddings.')
    parser.add_argument(
        '--relative_pos',
        action='store_true',
        help="""
            use relative position in the attention, instead of positional encoding.
            currently supports the simplest case: left (0), self(1), right(2)
            """)
    parser.add_argument(
        '--no_source_pos',
        action='store_true',
        help="remove source sentence orders")

    return parser


@register_config('model_multilingual')
def build_model_multilingual_config(parser):
    parser.add_argument(
        '--multi',
        action='store_true',
        help='enable multilingual training for Transformer.')

    parser.add_argument(
        '--sample_prob',
        nargs='*',
        type=float,
        help='probabilities of each input dataset.')
    parser.add_argument(
        '--local_attention',
        type=int,
        default=0,
        help='force to use local attention for the first K layers.')

    return parser


@register_config('model_insertion_transformer')
def build_model_insertion_transformer_config(parser):
    parser.add_argument(
        '--training_with_terminal',
        action='store_true',
        help='the model is trained to predict the <T> everywhere for empty.')
    parser.add_argument(
        '--parallel_decoding',
        action='store_true',
        help='use parallel decoding for Insertion Transformer')
    parser.add_argument(
        '--termination_penalty',
        type=float,
        default=0.0,
        help='use penalty to prevent model to terminate too early.')
    return parser


@register_config('model_transformer_indigo')
def build_model_transformer_indigo_cofig(parser):
    parser.add_argument('--insertable', action='store_true')
    parser.add_argument(
        '--insert_mode',
        choices=['word_first', 'position_first', 'balanced'],
        type=str,
        default='word_first')
    parser.add_argument(
        '--order',
        choices=['fixed', 'random', 'optimal', 'search_optimal', 'trainable'],
        type=str,
        default='fixed')
    parser.add_argument(
        '--path_temp',
        default=0,
        type=float,
        help='temperature to choose paths. 0 means choosing the top path only')
    parser.add_argument(
        '--beta',
        type=int,
        default=4,
        help='beam-size to search optimal paths.')
    parser.add_argument(
        '--ln_pos',
        action='store_true',
        help='a linear layer over the embedding and query.')
    parser.add_argument(
        '--no_bound', action='store_true', help='no boundary probabilitis.')
    parser.add_argument(
        '--no_weights',
        action='store_true',
        help='do not use reweighting after beam-search.')
    parser.add_argument(
        '--search_with_dropout',
        action='store_true',
        help='no boundary probabilitis.')

    parser.add_argument(
        '--epsilon',
        type=float,
        default=0,
        help='possibility to choose random order during training.')
    parser.add_argument(
        '--gamma', type=float, default=1, help='balance p(x) and p(z)')
    parser.add_argument(
        '--sample_order',
        action='store_true',
        help='perform sampling instead of beam-search')
    parser.add_argument(
        '--resampling',
        action='store_true',
        help='resampling after every samling operations')
    parser.add_argument(
        '--l2r_guided',
        action='store_true',
        help='using L2R order to guide the model a bit')
    parser.add_argument(
        '--adaptive_ess_ratio',
        default=0.5,
        type=float,
        help='th of adaptive effective sample size')
    parser.add_argument(
        '--use_gumbel',
        action='store_true',
        help='resampling after every samling operations')
    parser.add_argument(
        '--esteps',
        type=int,
        default=1,
        help='possibility to choose random order during training.')
    parser.add_argument(
        '--gsteps',
        type=int,
        default=1,
        help='possibility to choose random order during training.')
    parser.add_argument(
        '--decouple',
        action='store_true',
        help='decouple the scorer and the trainer. use best model.')

    return parser


@register_config('model_blockwise')
def build_model_blcokwise_config(parser):
    parser.add_argument(
        '--multi_width',
        type=int,
        default=1,
        help='default not use multi-step prediction')

    parser.add_argument(
        '--dyn',
        type=float,
        default=0.0,
        help='dynamic block-wse decoding (experimental)')
    parser.add_argument(
        '--random_path',
        action='store_true',
        help='use a random path(experimental)')
    parser.add_argument(
        '--exact_match',
        action='store_true',
        help='match with the 1-step model (experimental)')
    parser.add_argument('--constant_penalty', type=float, default=0)

    return parser


@register_config('model_bert')
def build_model_bert_config(parser):
    # support BERT models
    parser.add_argument(
        '--use_bert',
        type=str,
        default=None,
        choices=[
            "bert-base-uncased", "bert-large-uncased", "bert-base-cased",
            "bert-large-cased", "bert-base-multilingual-uncased",
            "bert-base-multilingual-cased", "bert-base-chinese"
        ])
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help='force to use lower case to process the sentence.')

    return parser
