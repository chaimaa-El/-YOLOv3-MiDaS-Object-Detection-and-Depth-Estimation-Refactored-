import torch
from torch import nn
from utils.google_utils import *
from model.blocks import FeatureFusionBlock, Interpolate, _make_encoder
from utils.parse_config import *
from utils import torch_utils

ONNX_EXPORT = False

class MDENet(nn.Module):
    """
    MDENet class combines YOLOv3 object detection with Midas depth estimation in a multi-feature, multi-scale network.
    """
    def __init__(self, yolo_props, path=None, freeze={}, features=256, non_negative=True, img_size=(416, 416), verbose=False):
        
        """
        Initialize MDENet model.
        
        Args:
            yolo_props (dict): Properties of YOLOv3, including anchors and number of classes.
            path (str, optional): Path to pre-trained model weights. Defaults to None.
            freeze (dict): Dictionary to freeze specific parts of the model (e.g., 'resnet', 'midas', 'yolo').
            features (int): Number of features in the encoder. Defaults to 256.
            non_negative (bool): If True, applies ReLU activation to ensure non-negative outputs. Defaults to True.
            img_size (tuple): Image size for the input (height, width). Defaults to (416, 416).
            verbose (bool): If True, prints additional information for debugging. Defaults to False.
        """
                
        super(MDENet, self).__init__()

        use_pretrained = True if path is None else False

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)
        
        
        # print(self.pretrained)
        
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )
        
        # YOLO head
        conv_output = (int(yolo_props["num_classes"]) + 5) * int((len(yolo_props["anchors"]) / 3))

        # Upsampling layers for the YOLO output at different scales
        self.upsample1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Layers for detecting small objects (YOLO1)
        self.yolo1_learner = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.yolo1_reduce = nn.Conv2d(1024, conv_output, kernel_size=1, stride=2, padding=1)
        self.yolo1 = YOLOLayer(yolo_props["anchors"][:3],
                               nc=int(yolo_props["num_classes"]),
                               img_size=img_size,
                               yolo_index=0,
                               layers=[],
                               stride=32)
        
        # Layers for detecting medium objects (YOLO2)
        self.yolo2_learner = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.yolo2_reduce = nn.Sequential(
            nn.Conv2d(512, conv_output, kernel_size=1, stride=2)
        )
        self.yolo2 = YOLOLayer(yolo_props["anchors"][3:6],
                               nc=int(yolo_props["num_classes"]),
                               img_size=img_size,
                               yolo_index=1,
                               layers=[],
                               stride=16)
        
        
        # Layers for detecting large objects (YOLO3)
        self.yolo3_learner = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        
        self.yolo3_reduce = nn.Sequential(
            nn.Conv2d(256, conv_output, kernel_size=1, stride=2)
        )
        self.yolo3 = YOLOLayer(yolo_props["anchors"][6:],
                               nc=int(yolo_props["num_classes"]),
                               img_size=img_size,
                               yolo_index=1,
                               layers=[],
                               stride=8)
        
        
        # Combine YOLO layers into a list for easy access
        self.yolo = [self.upsample1, self.upsample2, self.yolo1_learner, self.yolo1,
                     self.yolo2_learner, self.yolo2, self.yolo3_learner, self.yolo3,
                     self.yolo1_reduce, self.yolo2_reduce, self.yolo3_reduce]

        # Load pre-trained weights if a path is provided
        
        if path:
            self.load(path)
        
        # Freeze model parts 
        if freeze['resnet'] == "True":
            path = "freeze"
        if freeze['midas'] == "True":
            for param in self.scratch.parameters():
                param.requires_grad = False
        if freeze['yolo'] == "True":
            for mod in self.yolo:
                for param in mod.parameters():
                    param.requires_grad = False
        if freeze["planercnn"] == "True":
            pass
        
        if path is not None:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        

    def forward(self, x, augment=False, verbose=False):

        """
        Forward pass through the network. Performs depth estimation and object detection.

        Args:
            x (tensor): Input image tensor.
            augment (bool, optional): Whether to apply augmentations for testing. Defaults to False.
            verbose (bool, optional): If True, prints additional debugging information. Defaults to False.

        Returns:
            tuple: Contains depth estimates and YOLO detections.
        """
        
        if not augment:
            return self.forward_net(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y1 = []
            y2 = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                out = self.forward_net(xi)
                y1.append(out[0])
                y2.append(out[1])

            y2[1][..., :4] /= s[0]  # scale
            y2[1][..., 0] = img_size[1] - y2[1][..., 0]  # flip lr
            y2[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi
            
            #y1 = torch.cat(y1, 1)
            y2 = torch.cat(y2, 1)
            
            return y1, y2, None

    def forward_net(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)
            
        midas_out, yolo_out = self.run_batch(x)
        
        """
        for i, module in enumerate(self.module_list):
            # print("module", module)
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''
        """

        if self.training:  # train
            return midas_out, yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return midas_out, x, p
    
    def run_batch(self, x):
        
        """
        This function processes a batch of images `x` through the model and returns both depth and object detection outputs.
        
        Args:
            x (Tensor): Input batch of images, shape (batch_size, channels, height, width).
        
        Returns:
            tuple: A tuple containing:
                - depth_out (Tensor): The output of the depth detection network.
                - yolo_out (List[Tensor]): List of outputs from the YOLO object detection network.
        """
        
        # Pass input through the pretrained ResNet101 backbone to extract features
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        # Process the extracted features through custom layers for depth detection
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

     
        # Apply refinement networks to progressively refine depth maps
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Generate the final depth map
        depth_out = self.scratch.output_conv(path_1)
        
        # Object Detection (YOLO)
        # Pass through YOLO layers for object detection at different scales

        # Process layer_3 through the first YOLO detection head
        yolo1_out = self.yolo1(self.yolo1_reduce(self.yolo1_learner(layer_3)))
        
        # Upsample layer_3 and concatenate with layer_2 for the second detection head
        layer_3 = self.upsample1(layer_3)
        layer_2 = torch.cat([layer_3, layer_2], dim=1)
        layer_2 = self.yolo2_learner(layer_2)
        yolo2_out = self.yolo2(self.yolo2_reduce(layer_2))
        
        # Upsample layer_2 and concatenate with layer_1 for the third detection head
        layer_2 = self.upsample2(layer_2)
        layer_1 = torch.cat([layer_1, layer_2], dim=1)
        layer_1 = self.yolo3_learner(layer_1)
        yolo3_out = self.yolo3(self.yolo3_reduce(layer_1))
        
        # Collect YOLO outputs from all three scales
        yolo_out = [yolo1_out, yolo2_out, yolo3_out]
        
        return depth_out, yolo_out

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers throughout the model for optimized inference.
        This process reduces the number of layers and speeds up the model by combining 
        convolution and batch normalization layers.
        """
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        
        # Iterate over the model's children and fuse Conv2d and BatchNorm2d layers
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # Fuse batch normalization with previous convolution layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        
        # Update the module list with the fused layers
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # Print model info unless exporting to ONNX

    def info(self, verbose=False):
        """
        Displays the model's summary and details such as layer dimensions and number of parameters.
        
        Args:
            verbose (bool): Whether to display detailed layer information.
        """
        torch_utils.model_info(self, verbose)
    
    def load(self, path):
        """
        Loads model weights from a specified file path and applies them to the model.
        
        Args:
            path (str): The path to the file containing the saved model weights.
        """
        print("Loading model from:", path)
        parameters = torch.load(path, map_location=torch.device('cpu'))

        # If optimizer state is saved along with the model, load only the model weights
        if "optimizer" in parameters:
            parameters = parameters["model"]

        # Load the state dict (weights) into the model
        self.load_state_dict(parameters)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        """
        Initializes the YOLO detection layer which processes predictions for object localization 
        and classification. Each YOLO layer corresponds to a particular detection scale.
        
        Args:
            anchors (list): The anchor box dimensions used for bounding box predictions.
            nc (int): The number of object classes.
            img_size (tuple): The input image size (height, width).
            yolo_index (int): The index of this YOLO layer.
            layers (list): The indices of layers that are used for this YOLO detection.
            stride (int): The stride of the layer, which determines the scale of the detections.
        """
        super(YOLOLayer, self).__init__()
        
        # Initialize YOLO layer parameters
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.nl = len(layers)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5  # Output size: (classes + 4 bounding box coordinates + 1 objectness score)
        
        self.nx, self.ny, self.ng = 0, 0, 0  # Initialize grid size (nx, ny)
        self.anchor_vec = self.anchors / self.stride  # Adjust anchor boxes by stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)  # Reshape for later use

        if ONNX_EXPORT:
            self.training = False  # Disable training mode when exporting to ONNX
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # Create grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        """
        Creates the grid of (x, y) points for object detection. These grids are used to predict 
        the location of objects within each cell of the grid.
        
        Args:
            ng (tuple): Grid size (ny, nx).
            device (str): Device to store the grid on (e.g., 'cpu', 'cuda').
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng)
        
        if not self.training:
            # Create grid of xy offsets if not training
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            # Move anchor and grid tensors to the correct device (CPU or GPU)
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        """
        Performs the forward pass for the YOLO layer, making predictions for bounding box coordinates
        and objectness scores. The predictions are then processed to generate final object detection results.
        
        Args:
            p (Tensor): The input tensor containing the raw predictions from the model.
        
        Returns:
            tuple: The output of the YOLO layer, including:
                - p_cls (Tensor): The class probabilities for each bounding box.
                - xy (Tensor): The predicted bounding box coordinates (x, y).
                - wh (Tensor): The predicted bounding box width and height.
        """
        if ONNX_EXPORT:
            bs = 1  # Batch size for ONNX export
        else:
            bs, _, ny, nx = p.shape  # Get the batch size and grid size (ny, nx)
            if (self.nx, self.ny) != (nx, ny):
                # Recreate grid if the grid size changes
                self.create_grids((nx, ny), p.device)

        # Reshape predictions for YOLO output
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid = self.grid.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/mde.cfg', weights='weights/mde.weights'):
    # Converts between PyTorch and MDENet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = MDENet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if weights and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
