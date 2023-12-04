python

class Trainer:
    def __init__(self, hyp, opt, device, tb_writer=None):
        self.hyp = hyp
        self.opt = opt
        self.device = device
        self.tb_writer = tb_writer
        self.logger = logging.getLogger(__name__)
        self.save_dir = Path(opt.save_dir)
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.total_batch_size = opt.total_batch_size
        self.weights = opt.weights
        self.rank = opt.global_rank
        self.freeze = opt.freeze
        self.wdir = self.save_dir / 'weights'
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last = self.wdir / 'last.pt'
        self.best = self.wdir / 'best.pt'
        self.results_file = self.save_dir / 'results.txt'
        self.plots = not opt.evolve  # create plots
        self.cuda = device.type != 'cpu'
        self.init_seeds(2 + self.rank)
        with open(opt.data) as f:
            self.data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        self.is_coco = opt.data.endswith('coco.yaml')
        self.loggers = {'wandb': None}  # loggers dict
        if self.rank in [-1, 0]:
            self.opt.hyp = self.hyp  # add hyperparameters
            run_id = torch.load(self.weights, map_location=self.device).get('wandb_id') if self.weights.endswith('.pt') and os.path.isfile(self.weights) else None
            wandb_logger = WandbLogger(self.opt, Path(self.opt.save_dir).stem, run_id, self.data_dict)
            self.loggers['wandb'] = wandb_logger.wandb
            self.data_dict = wandb_logger.data_dict
            if wandb_logger.wandb:
                self.weights, self.epochs, self.hyp = self.opt.weights, self.opt.epochs, self.opt.hyp  # WandbLogger might update weights, epochs if resuming
        self.nc = 1 if self.opt.single_cls else int(self.data_dict['nc'])  # number of classes
        self.names = ['item'] if self.opt.single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']  # class names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt.data)  # check
        self.pretrained = self.weights.endswith('.pt')
        if self.pretrained:
            with torch_distributed_zero_first(self.rank):
                attempt_download(self.weights)  # download if not found locally
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
            self.model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
            exclude = ['anchor'] if (self.opt.cfg or self.hyp.get('anchors')) and not self.opt.resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), self.weights))  # report
        else:
            self.model = Model(self.opt.cfg, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)  # check
        self.train_path = self.data_dict['train']
        self.test_path = self.data_dict['val']
        self.freeze = [f'model.{x}.' for x in (self.freeze if len(self.freeze) > 1 else range(self.freeze[0]))]  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in self.freeze):
                print('freezing %s' % k)
                v.requires_grad = False
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.total_batch_size * self.accumulate / self.nbs  # scale weight_decay
        self.logger.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")
        self.pg0, self.pg1, self.pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                self.pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                self.pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                self.pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):           
                    self.pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        self.pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):           
                    self.pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        self.pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):           
                    self.pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        self.pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):           
                    self
