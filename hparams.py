import tensorflow as tf

# Default hyperparameters:



class Hparams:
    def __init__(self, cleaners='english_cleaners',

                 # Audio:
                 num_mels=80,
                 num_freq=1025,
                 sample_rate=16000,
                 frame_length_ms=50,
                 frame_shift_ms=12.5,
                 preemphasis=0.97,
                 min_level_db=-100,
                 ref_level_db=20,

                 # Model:
                 outputs_per_step=2,
                 embed_depth=256,
                 prenet_depths=[256, 128],
                 encoder_depth=256,
                 rnn_depth=256,

                 # Attention
                 attention_depth=256,

                 # Training:
                 batch_size=32,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 initial_learning_rate=0.002,
                 decay_learning_rate=True,
                 use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

                 # Eval:
                 max_iters=1000,
                 griffin_lim_iters=60,
                 power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim

                 # Global style token
                 use_gst=True,
                 # When false, the scripit will do as the paper  "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
                 num_gst=10,
                 num_heads=4,  # Head number for multi-head attention
                 style_embed_depth=256,
                 reference_filters=[32, 32, 64, 64, 128, 128],
                 reference_depth=128,
                 style_att_type="mlp_attention",
                 # Attention type for style attention module (dot_attention, mlp_attention)
                 style_att_dim=128):
        self.cleaners=cleaners

        # Audio:
        self.num_mels =num_mels
        self.num_freq = num_freq
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.preemphasis = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

        # Model:
        self.outputs_per_step=outputs_per_step
        self.embed_depth = embed_depth
        self.prenet_depths = prenet_depths
        self.encoder_depth = encoder_depth
        self.rnn_depth = rnn_depth

        # Attention
        self.attention_depth = attention_depth

        # Training:
        self.batch_size = batch_size
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.initial_learning_rate = initial_learning_rate
        self.decay_learning_rate = decay_learning_rate
        self.use_cmudict = use_cmudict  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

        # Eval:
        self.max_iters = max_iters
        self.griffin_lim_iters = griffin_lim_iters
        self.power = power  # Power to raise magnitudes to prior to Griffin-Lim

        # Global style token
        self.use_gst = use_gst  # When false, the scripit will do as the paper  "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
        self.num_gst = num_gst
        self.num_heads = num_heads  # Head number for multi-head attention
        self.style_embed_depth = style_embed_depth
        self.reference_filters = reference_filters
        self.reference_depth = reference_depth
        self.style_att_type = style_att_type  # Attention type for style attention module (dot_attention, mlp_attention)
        self.style_att_dim = style_att_dim

hparams = Hparams()

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)

