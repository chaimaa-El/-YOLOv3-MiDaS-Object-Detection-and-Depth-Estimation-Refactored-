import argparse
from sys import platform
import configparser as cp
from model.mde_net import *  # Import the depth estimation model (MDENet)
from model.monocular_depth_estimation_dataset import *  # Load datasets for monocular depth estimation
from utils.utils import *  # Utility functions for image processing, model handling, etc.

# Read configuration settings from a file
conf = cp.RawConfigParser()
conf_path = "cfg/mde.cfg"  # Path to the config file
conf.read(conf_path)

# Set YOLO model properties from the config
yolo_props = {}
yolo_props["anchors"] = np.array([float(x) for x in conf.get("yolo", "anchors").split(',')]).reshape((-1, 2))
yolo_props["num_classes"] = conf.get("yolo", "classes")

# Set freeze and alpha values for different model components based on the config
freeze = {}
alpha = {}
freeze["resnet"], alpha["resnet"] = (True, 0) if conf.get("freeze", "resnet") == "True" else (False, 1)
freeze["midas"], alpha["midas"] = (True, 0) if conf.get("freeze", "midas") == "True" else (False, 1)
freeze["yolo"], alpha["yolo"] = (True, 0) if conf.get("freeze", "yolo") == "True" else (False, 1)
freeze["planercnn"], alpha["planercnn"] = (True, 0) if conf.get("freeze", "planercnn") == "True" else (False, 1)

def detect(save_img=False):
    """
    Perform object detection and depth estimation on input images or video streams.
    The function runs inference using the configured model, applies NMS, and optionally saves results.
    
    :param save_img: Flag to indicate whether to save output images
    """
    # Define the image size for inference (320x192 if ONNX export is enabled, otherwise use config size)
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Select device (CPU or CUDA-enabled GPU)
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    
    # Remove and create the output directory
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # Initialize the model for monocular depth estimation
    model = MDENet(path=opt.weights, yolo_props=yolo_props, freeze=freeze).to(device)

    # Set the model to evaluation mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers for optimization (only supported if model allows it)
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # Placeholder image tensor for ONNX export
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # Export file path
        torch.onnx.export(model, img, f, verbose=False, opset_version=11)

        # Validate ONNX model
        import onnx
        model = onnx.load(f)
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))
        return

    # Enable half precision (FP16) if device supports it and the option is enabled
    half = half and device.type != 'cpu'
    if half:
        model.half()

    # Initialize data loader based on the input source (webcam or image files)
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # Optimize performance for constant image size
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Load class names and assign random colors for bounding boxes
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Start inference
    t0 = time.time()
    _ = model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # Warm-up
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert to float or half precision
        img /= 255.0  # Normalize image to range [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Run model inference (depth and object detection predictions)
        t1 = torch_utils.time_synchronized()
        depth_pred, pred, _ = model(img, augment=False)
        t2 = torch_utils.time_synchronized()

        # Convert predictions to float if using half precision
        if half:
            pred = pred.float()

        # Apply Non-Maximum Suppression (NMS) to filter out weak detections
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process each detection and apply necessary transformations
        for i, det in enumerate(pred):
            if webcam:  # Handle batch processing if webcam is used
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # Print image size
            if det is not None and len(det):
                # Rescale detection boxes to the original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print detected classes and their counts
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])

                # Save detection results and draw bounding boxes
                for *xyxy, conf, cls in det:
                    if save_txt:  # Save detections to text file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Draw bounding boxes on the image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print inference time for each detection
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Display the image with bounding boxes (if applicable)
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # Exit on 'q'
                    raise StopIteration

            # Save detection results (image or video)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # New video file
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

        # Save depth predictions
        for i, depth in enumerate(depth_pred):
            prediction = depth.squeeze().cpu().numpy()
            splt = save_path.split(".")
            filename = splt[0] + "_depth." + splt[1]
            print("depth_path", filename)
            write_depth(filename, prediction, bits=2)

    # Print saved results information
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS specific
            os.system('open ' + out + ' ' + save_path)

    # Print total execution time
    print('Done. (%.3fs)' % (time.time() - t0))


# Command-line argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-custom.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/customdata/custom.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='weights path')
    parser.add_argument('--source', type=str, default='data/images', help='source path for images/videos')
    parser.add_argument('--output', type=str, default='output', help='output directory')
    parser.add_argument('--device', default='', help='CUDA device')
    parser.add_argument('--view_img', action='store_true', help='show results')
    parser.add_argument('--save_img', action='store_true', help='save results')
    parser.add_argument('--save_txt', action='store_true', help='save results to txt file')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold for object detection')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='Intersection over Union threshold for NMS')
    parser.add_argument('--half', action='store_true', help='use half precision FP16')
    parser.add_argument('--agnostic_nms', action='store_true', help='agnostic class NMS')
    parser.add_argument('--classes', type=int, nargs='+', help='filter by class')

    opt = parser.parse_args()
    detect()
