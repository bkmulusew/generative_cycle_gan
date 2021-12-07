import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class GenerativeCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    #DONE
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_reconstruction', type=float, default=1.0, help='weight for the reconstruction loss')  # You can define new arguments for this model.
            parser.add_argument('--lambda_discrimination', type=float, default=1.5, help='weight for the discriminator loss')  # You can define new arguments for this model.
            parser.add_argument('--lambda_encoder', type=float, default=0, help='weight for the encoder loss')  # You can define new arguments for this model.
        return parser


    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Encoder_Loss', 'Autoencoder_Loss', 'Discriminator_Loss', 'Total_Loss']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['Input_Images', 'GAN_Images', 'Encoded_Images', 'Decoded_Images']
        else:
            self.visual_names = ['Input_Images', 'GAN_Images', 'Encoded_Images']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A', 'Encoder', 'Decoder']
            self.pretrainedGAN_models = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'Encoder']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netEncoder = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDecoder = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.reconstructionLoss = torch.nn.L1Loss()
            self.criterionCE = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_Encoder = torch.optim.Adam(self.netEncoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Decoder = torch.optim.Adam(self.netDecoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Encoder)
            self.optimizers.append(self.optimizer_Decoder)


    #DONE
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.Input_Images = input['A' if AtoB else 'B'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    #DONE
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.GAN_Images = self.netG_A(self.Input_Images)  # G_A(A)
        self.Encoded_Images = self.netEncoder(self.GAN_Images)   # G_B(G_A(A))

        if self.isTrain:
            self.Decoded_Images = self.netDecoder(self.Encoded_Images)

    def backward(self):
        self.loss_Encoder_Loss = self.criterionCE(self.GAN_Images, self.Encoded_Images) * self.opt.lambda_encoder
        self.loss_Autoencoder_Loss = self.reconstructionLoss(self.Decoded_Images, self.GAN_Images) * self.opt.lambda_reconstruction
        self.loss_Discriminator_Loss = self.criterionGAN(self.netD_A(self.Encoded_Images), True) * self.opt.lambda_discrimination
        self.loss_Total_Loss = self.loss_Discriminator_Loss + self.loss_Autoencoder_Loss - self.loss_Encoder_Loss
        self.loss_Total_Loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netG_A, self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_Encoder.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_Decoder.zero_grad()
        self.backward()
        self.optimizer_Encoder.step()  # set G_A and G_B's gradients to zero
        self.optimizer_Decoder.step()




