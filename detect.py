# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
from sort.sort import Sort
import torch

#路径获取部分
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#导入yolo辅助模块，以进行检测和识别任务
#通用的函数和类，比如图像的处理、非极大值抑制等
from models.common import DetectMultiBackend
#这个文件定义了两个类，LoadImages和LoadStreams，它们可以加载图像或视频帧，并对它们进行一些预处理，以便进行物体检测或识别。
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
#这个文件定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等。
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
#这个文件定义了Annotator类，可以在图像上绘制矩形框和标注信息。
from utils.plots import Annotator, colors, save_one_box
#这个文件定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等等。
from utils.torch_utils import select_device, smart_inference_mode


def box_iou(box1, box2):#新增的
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list or tuple): The first bounding box, specified as [x1, y1, x2, y2].
        box2 (list or tuple): The second bounding box, specified as [x1, y1, x2, y2].

    Returns:
        float: The IoU of the two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


@smart_inference_mode()
#定义run()函数，括号中首先设置参数
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        # source=ROOT / 'data/ships',  # file/dir/URL/glob/screen/0(webcam)
        source=ROOT / 'data/ships/20140521222816_Compress.AVI',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/myships.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    #根据输入的source确定输入数据的类型，以及是否需要保存输出结果
    #首先将source转换为字符串类型，然后判断是否需要保存输出结果。如果nosave和source的后缀不是.txt，则会保存输出结果
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    #接着根据source的类型，确定输入数据的类型。如果source的后缀是图像或视频格式之一，那么将is_file设置为True；如果source以rtsp://、rtmp://、http://、https://开头，那么将is_url设置为True；如果source是数字或以.txt结尾或是一个URL，那么将webcam设置为True。如果source既是文件又是URL，那么会调用check_file函数下载文件。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    #新的，估计是实时捕获屏幕截图的功能
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    #这段代码主要是用于创建保存输出结果的目录。
    # 首先将project和name拼接成完整路径，并且使用increment_path函数来确保目录不存在，如果存在，则在名称后面添加递增的数字。然后在这个目录下创建labels子目录（如果save_txt为True），用于保存输出结果的标签文件，否则创建一个空的目录用于保存输出结果。这个过程中，如果目录已经存在，而exist_ok为False，那么会抛出一个异常，指示目录已存在。如果exist_ok为True，则不会抛出异常，而是直接使用已经存在的目录。
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #这段代码主要是用于选择设备、初始化模型和检查图像大小。
    # 首先调用select_device函数选择设备，如果device为空，则使用默认设备。然后使用DetectMultiBackend类来初始化模型，其中weights是指模型的权重路径，device是指设备，dnn是指是否使用OpenCV DNN，data是指数据集配置文件的路径，fp16是指是否使用半精度浮点数进行推理。接着从模型中获取stride、names和pt等参数，其中stride是指下采样率，names是指模型预测的类别名称，pt是指PyTorch模型对象。最后调用check_img_size函数检查图像大小是否符合要求，如果不符合则进行调整。
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    #这里是根据输入的 source 参数来判断是否是通过 webcam 摄像头捕捉视频流，如果是则使用 LoadStreams 加载视频流，否则使用 LoadImages 加载图像。如果是 webcam 模式，则设置 cudnn.benchmark = True 以加速常量图像大小的推理。bs 表示 batch_size（批量大小），这里是 1 或视频流中的帧数。vid_path 和 vid_writer 分别是视频路径和视频编写器，初始化为长度为 batch_size 的空列表。
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    #vid_writer好像就是一个列表，每一个元素就是一张图，对视频帧逐张处理
#这里读取了数据

    #模型热身，即对模型进行一些预处理以加速后续的推理过程。代码中首先定义了一些变量，包括seen、windows和dt，分别表示已处理的图片数量、窗口列表和时间消耗列表。接着对数据集中的每张图片进行处理，首先将图片转换为Tensor格式，并根据需要将其转换为FP16或FP32格式。然后将像素值从0-255转换为0.0-1.0，并为批处理增加一维。最后记录时间消耗并更新dt列表。
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
#在进循环前初始化实例Sort
    mot_tracker = Sort() #create instance of the SORT tracker
#-----------------------------------------------------------------------------------------------------------------------------------------------↓这里是一帧的循环
####这个for循环，每跑一次就走一帧
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
       
        #这段代码似乎与使用计算机视觉模型进行预测有关。第一行代码创建了一个名为“visualize”的变量，如果需要可视化，则将其设置为保存可视化结果的路径，否则将其设置为False。使用increment_path函数创建路径，如果文件名已存在，则将数字附加到文件名后面以避免覆盖已有文件。
        # 第二行代码使用model函数对图像im进行预测，augment和visualize参数用于指示是否应该在预测时使用数据增强和可视化。
        #第三行代码记录了当前时间，并计算从上一个时间点到这个时间点的时间差，然后将这个时间差加到一个名为dt的时间差列表中的第二个元素上。
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        #执行非最大值抑制（NMS）的步骤，用于筛选预测结果。
        # non_max_suppression函数的输入参数包括预测结果pred、置信度阈值conf_thres、IOU（交并比）阈值iou_thres、类别classes、是否进行类别无关的NMSagnostic_nms，以及最大检测数max_det。该函数的输出是经过NMS筛选后的预测结果。
        # 补充一下:agnostic-nms是跨类别nms,比如待检测图像中有一个长得很像排球的足球,pt文件的分类中有足球和排球两种,那在识别时这个足球可能会被同时框上2个框:一个是足球,一个是排球。agnostic-nms：是否使用类别不敏感的非极大抑制（即不考虑类别信息），默认为 False
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 这段代码使用了一个循环来遍历检测结果列表中的每个物体，并对每个物体进行处理。循环中的变量"i"是一个索引变量，表示当前正在处理第几个物体，而变量"det"则表示当前物体的检测结果。循环体中的第一行代码 "seen += 1" 用于增加一个计数器，记录已处理的物体数量。
        # 接下来，代码会根据是否使用网络摄像头来判断处理单张图像还是批量图像。如果使用的是网络摄像头，则代码会遍历每个图像并复制一份备份到变量"im0"中，同时将当前图像的路径和计数器记录到变量"p"和"frame"中。最后，代码会将当前处理的物体索引和相关信息记录到字符串变量"s"中。
        # 如果没有使用网络摄像头，则代码会直接使用"im0s"变量中的图像，将图像路径和计数器记录到变量"p"和"frame"中。同时，代码还会检查数据集中是否有"frame"属性，如果有，则将其值记录到变量"frame"中。
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 这段代码中，首先将图像路径转换为"Path"对象。接下来，代码使用"save_dir"变量中的路径和图像文件名来构建保存检测结果图像的完整路径，并将其保存在变量"save_path"中。代码还会根据数据集的模式（"image"或"video"）来构建保存检测结果标签的文件路径，并将其保存在变量"txt_path"中。
            # 在处理图像路径和文件路径之后，代码会将图像的尺寸信息添加到字符串变量"s"中，以便于打印。接着，代码会计算归一化增益"gn"，并将其保存在变量中，以便后续使用。接下来，代码会根据是否需要保存截取图像的标志"save_crop"来选择是否要对原始图像进行复制，以备保存截取图像时使用。最后，代码创建了一个"Annotator"对象，以便于在图像上绘制检测结果。
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()###此时，det.real中存放的为tensor([[122,  37, 366, 140， 0.96800,1]])
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#看看放哪儿去何时
                # mot_tracker = Sort()    #找到一个，dataset.frame可以指示，如果=0说明是图片，如果是视频就是从1到n   还有个det，应该是运行完调用的检测函数之后在别的.py里出现的结果变量。
                trackers = mot_tracker.update(det)  #det=tensor([[167.00000,  74.00000, 326.00000, 123.00000,   0.87564,   1.00000]]),xyxy score c 
                for d in trackers:
                    print(d)

#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
# ####很妙，xyxy转成xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
# ####自定义label输出格式
#                         with open(f'{txt_path}.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                         label = None if hide_labels else (f'{names[c]} {conf:.2f} ID:{d[-1]}' if not hide_conf else f'{names[c]} ID:{d[-1]}')
# #要在这儿之前把sort引入，用xyxy,c,
#                         annotator.box_label(xyxy, label, color=colors(c, True))
#                         #这里应该就是往图上画框的过程！
# ####对这个结果进行处理，信息都齐了，c=1为类别class，label='ship 0.97'为概率，xyxy=[tensor(75.), tensor(64.), tensor(437.), tensor(228.)]为框的xyxy,如果要xyxy转成xywh就xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                for *xyxy, conf, cls in reversed(det):#sage写的，我也不知道对不对
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None
                        for d in trackers:
                            if box_iou(xyxy, d[:4]) > iou_thres:
                                label = names[c]
                                if not hide_conf:
                                    label += f' {conf:.2f}'
                                label += f' ID:{d[-1]}'
                                break
                        annotator.box_label(xyxy, label, color=colors(c, True))



                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                #我猜上面这个循环是一个帧内


            # 如果需要在窗口中实时查看检测结果，则代码会使用OpenCV库中的函数将图像显示在窗口中，并等待1毫秒以便继续下一帧的检测。代码会检查是否已经为当前图像创建了窗口（if p not in windows），并在必要时创建窗口，并使用图像名称来命名该窗口。窗口的名称是由变量"p"指定的图像路径名。如果检测到图像尚未在窗口中打开，则代码会创建一个新窗口并将图像显示在窗口中。如果图像已经在窗口中打开，则代码会直接更新窗口中的图像。       
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


            # ###这一段代码是一个目标检测算法中的推理过程，通过对一张或多张图片中的物体进行检测，输出检测结果，并将检测结果保存到文件或显示在窗口中。以下是每个步骤的详细说明：

            # 对于每个输入图片，将其路径、原始图像和当前帧数（如果存在）分别赋值给p、im0和frame变量；
            # 如果webcam为True，则将输出信息字符串s初始化为空，否则将其初始化为该数据集的“frame”属性；
            # 将p转换为Path类型，并生成保存检测结果的路径save_path和文本文件路径txt_path；
            # 将im0大小与目标检测的输入大小匹配，将检测结果det中的边界框坐标从img_size缩放到im0大小，然后将结果打印在输出字符串s中；
            # 如果save_txt为True，则将结果写入文本文件中；
            # 如果save_img、save_crop或view_img中任意一个为True，则将检测结果添加到图像中，并在窗口中显示结果；
            # 如果save_img为True，则保存结果图像；
            # 如果是视频数据集，则将结果写入视频文件中；
            # 最后，打印每个图片的检测时间。

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
#----------------------------------------------------------------------------↑这里是一帧的循环

    # 这部分代码用于输出检测结果和计算检测速度。
    # 首先，将检测得到的边界框（det）从img_size大小缩放到im0大小。然后，对于每个类别c，统计检测到的框的个数n，将其加入输出字符串s。
    # 接着，对于每个框，可以选择将其保存到txt文件中（若save_txt=True），并将其在图像中绘制出来。如果save_crop为True，则将该框对应的图像裁剪出来并保存。如果view_img为True，则在窗口中显示检测结果。最后，如果save_img为True，则将检测结果保存到文件中（可以是图片或视频）。
    # 输出结果包括每张图片的预处理、推理和NMS时间，以及结果保存的路径。如果update为True，则将模型更新，以修复SourceChangeWarning。
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/ships', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/ships', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/myships.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

# check_requirements(exclude=('tensorboard', 'thop')) 检查程序所需的依赖项是否已安装。
# run(**vars(opt)) 将 opt 变量的属性和属性值作为关键字参数传递给 run() 函数。
# opt = parse_opt() 解析命令行参数并将其存储在 opt 变量中。
# main(opt) 调用主函数，并将 opt 变量作为参数传递给它
def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
