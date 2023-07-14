import torch

def _create(name, pretrained=True, channels=3, classes=2, autoshape=True, verbose=True, device=None):

    from pathlib import Path
    from models.common import AutoShape, DetectMultiBackend
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device
    from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(ROOT / 'requirements.txt', exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # checkpoint path

    try:
        device = select_device(device)
        model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model 在models/common.py里的model检测模型！ 1 这两行居然缺一不可
        #⚠️⚠️把这个.py 函数 方法换成别的里的model应该就能实现换方法
        #不行，common.py里只能处理图片，视频得从其他检测方法里调用
        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS  # detection model 在models/common.py里的model检测模型！ 2这两行居然缺一不可
       
        # stride, names, pt = model.stride, model.names, model.pt
        # imgsz = check_img_size(imgsz, s=stride)  # check image size
        
        
        
        
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)
    except Exception as e:
        help_url = 'https://github.com/cytosineXT/yolov5_newnewship'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e

def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)

def yolov5s(pretrained=True, channels=3, classes=2, autoshape=True, _verbose=True, device=None):
    return _create('yolov5s', pretrained, channels, classes, autoshape, _verbose, device)