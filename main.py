from __future__ import print_function

import argparse

import cv2
from arc_face import *
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

from yoloV5_face.utils import google_utils
from yoloV5_face.utils.datasets import *
from yoloV5_face.utils.utils import *

from silentFace_model.predict_net import *
from silentFace_model.predict_net import AntiSpoofPredict

import sys

sys.path.append("./yoloV5_face")


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size, expected_size
    scale = min(eh / ih, ew / iw)  # Resize the largest side to 416
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2  # The height of the upper part to be filled
    bottom = eh - nh - top  # The height of the lower part to be filled
    left = (ew - nw) // 2  # The distance of the left part to be filled
    right = ew - nw - left  # The distance of the right part to be filled
    # Border filling
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


def cosin_metric(x1, x2):
    # Calculate the cosine distance
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    # image = cv2_letterbox_image(image,128)
    image = cv2.resize(image, (128, 128))
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featuresdict(model, dir):
    """
        Extracts features from images in a directory using the provided model.

        Args:
            model: The model used for feature extraction.
            dir (str): The directory containing the images.

        Returns:
            dict: A dictionary where keys are image filenames and values are corresponding feature vectors.
    """
    list = os.listdir(dir)
    person_dict = {}
    for i, each in enumerate(list):
        image = load_image(f"pic/{each}")
        data = torch.from_numpy(image)
        data = data.to(torch.device("cpu"))  # cpu or cuda
        output = model(data)  # Get features
        output = output.data.cpu().numpy()

        # Concatenate features to form a single feature vector
        fe_1 = output[0]
        fe_2 = output[1]
        # print("this",cnt)
        # print(fe_1.shape,fe_2.shape)
        feature = np.hstack((fe_1, fe_2))
        # print(feature.shape)

        person_dict[each] = feature
    return person_dict


def detect(opt, save_img=False):
    """
        Perform object detection on images or videos.

        Args:
            save_img (bool): Whether to save the images with detections.

        Returns:
            None
    """
    # Load configuration information
    out, source, yoloV5_weights, Areface_weights, silentFace_weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.yoloV5_weights, opt.Areface_weights, opt.silentFace_weights, opt.view_img, opt.save_txt, opt.img_size

    # Determine the type of test resource
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith(
        '.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    dir = "pic"

    # Create output folder
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # Whether the gpu supports half precision to improve performance
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model

    model = torch.load(yoloV5_weights, map_location=device)['model'].float()  # load to FP32

    model.to(device).eval()

    arcface_model = resnet_face18(False)

    # Load Arcface model to predict
    arcface_model = DataParallel(arcface_model)
    # load_model(model, opt.test_model_path)
    # arcface_model.load_state_dict(torch.load(Areface_weights), strict=False) # use it for gpu
    arcface_model.load_state_dict(torch.load(Areface_weights, map_location='cpu'), strict=False)  # use it for cpu
    arcface_model.to(device).eval()

    # Load Anti Sproof model to check video or image is fake or real
    pred_model = AntiSpoofPredict(0)

    if half:
        model.half()  # to FP16

    features = get_featuresdict(arcface_model, dir)

    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)

    else:
        # Load images and videos
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    view_img = True
    # Get names and colors Obtain the class names and colors of the boxes
    names = model.names if hasattr(model, 'names') else model.modules.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # Run inference Inference process
    t0 = time.time()

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # Data preprocessing
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS Perform nms to filter the boxes
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain wh wh

            if det is not None and len(det):  # If there are predicted targets (boxes exist)
                # Rescale boxes from img_size to im0 size Restore the position of the boxes on the original image
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results Print box result information
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:  # x1 y1 x2 y2 cls class
                    prediction = np.zeros((1, 3))
                    # crop
                    face_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    # rf_size
                    rf_img = cv2.resize(face_img, (80, 80))

                    # recognition
                    face_img = cv2.resize(face_img, (128, 128))

                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    face_img = np.dstack((face_img, np.fliplr(face_img)))

                    face_img = face_img.transpose((2, 0, 1))
                    face_img = face_img[:, np.newaxis, :, :]

                    face_img = face_img.astype(np.float32, copy=False)
                    face_img -= 127.5
                    face_img /= 127.5

                    face_data = torch.from_numpy(face_img)
                    face_data = face_data.to(device)

                    _output = arcface_model(face_data)  # Get features
                    _output = _output.data.cpu().numpy()

                    fe_1 = _output[0]
                    fe_2 = _output[1]

                    _feature = np.hstack((fe_1, fe_2))

                    label = "Person"
                    list = os.listdir(dir)
                    max_f = 0
                    t = 0
                    for i, each in enumerate(list):
                        t = cosin_metric(features[each], _feature)
                        # print(each, t)
                        if t > max_f:
                            max_f = t
                            max_n = each
                        # print(max_n,max_f)
                        if (max_f > 0.5):
                            label = f'{max_n[:-4]}_{max_f:.4f}'

                    if opt.open_rf:
                        # pred real or fake
                        for model_name in os.listdir(silentFace_weights):
                            prediction += pred_model.predict(rf_img, os.path.join(silentFace_weights, model_name))
                        rf_label = np.argmax(prediction)
                        value = prediction[0][rf_label] / 2
                        print(rf_label, value)
                        if rf_label == 1:
                            # label += "_real"
                            label = f"{label}"
                            if label.startswith("Person"):
                                label = "Person"
                                box_color = (0, 0, 255)  # Red for person
                            else:
                                box_color = (255, 0, 0)  # Blue for real face
                        else:
                            label = f"{label}_fake"
                            box_color = (0, 0, 255)  # Red for fake face
                    plot_one_box(xyxy, im0, label=label, color=box_color, line_thickness=3)  # colors[int(cls)]

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))  # Output execution time

            # Stream results
            # Display output
            if view_img:
                window_name = "Webcam"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # create a resizeable popup
                cv2.resizeWindow(window_name, 800, 600)  # resize popup (width, height)
                cv2.imshow(window_name, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    # raise StopIteration
                    break

            # Save results (image with detections)
            if save_img:  # Save pictures or videos
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        save_path = os.path.join(os.getcwd(), out)
        print('Results saved to', save_path)

        # Open file explorer window for the saved file (Windows only)
        if os.name == 'nt':  # Windows
            os.startfile(save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloV5_weights', type=str, default=r'weights/yolo5_weights/best.pt', help='yolov5.pt path')
    parser.add_argument('--Areface_weights', type=str, default=r'weights/arcface_weights/resnet18_110.pth',
                        help='arcface.pth path')
    parser.add_argument('--silentFace_weights', type=str, default=r"weights/anti_spoof_models/",
                        help='manti_spoof_models dir path')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--open_rf', default=1, help='if open real/fake 1 0 ')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect(opt)
