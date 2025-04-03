from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.t2m_bigru_vec263 import *
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

def load_bert():
    model_path = 'distilbert/distilbert-base-uncased'
    bert = BertTextEncoder(model_path)
    bert.eval()
    bert.text_model.training = False
    for name, param in bert.named_parameters():
        if 'bert_output_fc' in name:
            param.requires_grad = True  # 确保 mlp_fc.detail_description_mlp 的参数可训练
        else:
            param.requires_grad = False
    return bert


# our version
def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose']-4, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    if(opt['eval_encoder'] == 'bert'):
        text_enc = load_bert()
    else:
        text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                    pos_size=opt['dim_pos_ohot'],
                                    hidden_size=opt['dim_text_hidden'],
                                    output_size=opt['dim_coemb_hidden'],
                                    device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc



# our version
def build_evaluators_new(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose']-4, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], 't2m', "Decomp_SP001_SM001_H512", 'model', 'latest.tar'),
                            map_location=opt["device"])
    movement_enc.load_state_dict(checkpoint['movement_enc'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                    pos_size=opt['dim_pos_ohot'],
                                    hidden_size=opt['dim_text_hidden'],
                                    output_size=opt['dim_coemb_hidden'],
                                    device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    textencoder_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "t2m_textencoder":
            name = k.replace("t2m_textencoder.", "")
            textencoder_dict[name] = v
    text_enc.load_state_dict(textencoder_dict, strict=True)

    motionencoder_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "t2m_motionencoder":
            name = k.replace("t2m_motionencoder.", "")
            motionencoder_dict[name] = v
    motion_enc.load_state_dict(motionencoder_dict, strict=True)
    print("loading from t2m model!")
    return text_enc, motion_enc, movement_enc


# our wrapper
class EvaluatorMDMWrapper(object):

    def __init__(self,args, dataset_name, device):
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263 if dataset_name == 'humanml' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': '.',
            'unit_length': 4,
            'eval_encoder': args.eval_encoder
        }

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluators(opt)
        # self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluators_new(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens, caption):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
            '''Text Encoding'''
            if(self.opt['eval_encoder'] == 'bert'):
                text_embedding = self.text_encoder(caption)
            else:
                text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
    

# our wrapper
class EvaluatorAgnosticMDMWrapper(object):

    def __init__(self, dataset_name, device):
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263 if dataset_name == 'humanml' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': '.',
            'unit_length': 4
        }

        # self.text_encoder, self.motion_encoder = build_agnostic_evaluators(opt)
        self.text_encoder, self.motion_encoder = load_pretrained(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens, caption):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            '''Movement Encoding'''
            motion_embedding = self.motion_encoder(motions).loc 
            '''Text Encoding'''
            text_embedding = self.text_encoder(caption).loc 
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            motion_embedding = self.motion_encoder(motions).loc
        return motion_embedding


def build_agnostic_evaluators(opt):
    motion_enc=ActorAgnosticEncoder(nfeats=263, vae=True, num_layers=4)
    text_enc=DistilbertActorAgnosticEncoder( modelpath='distilbert/distilbert-base-uncased', num_layers=4)
    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc

def load_pretrained(opt):
    motion_enc=ActorAgnosticEncoder(nfeats=263, vae=True, num_layers=4)
    text_enc=DistilbertActorAgnosticEncoder( modelpath='distilbert/distilbert-base-uncased', num_layers=4)
    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    textencoder_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "textencoder":
            name = k.replace("textencoder.", "")
            textencoder_dict[name] = v
    text_enc.load_state_dict(textencoder_dict, strict=True)

    motionencoder_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "motionencoder":
            name = k.replace("motionencoder.", "")
            motionencoder_dict[name] = v
    motion_enc.load_state_dict(motionencoder_dict, strict=True)

    print("T2M Evaluator Model Loaded!")
    return text_enc, motion_enc


def load_pretrained_original(opt):
    motion_enc=ActorAgnosticEncoder(nfeats=263, vae=True, num_layers=4)
    text_enc=DistilbertActorAgnosticEncoder( modelpath='distilbert/distilbert-base-uncased', num_layers=4)
    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    textencoder_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "textencoder":
            name = k.replace("textencoder.", "")
            textencoder_dict[name] = v
    text_enc.load_state_dict(textencoder_dict, strict=True)

    motionencoder_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "motionencoder":
            name = k.replace("motionencoder.", "")
            motionencoder_dict[name] = v
    motion_enc.load_state_dict(motionencoder_dict, strict=True)

    print("T2M Evaluator Model Loaded!")
    return text_enc, motion_enc