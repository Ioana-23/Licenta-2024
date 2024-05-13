import os
import random
from dataclasses import dataclass

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as fn
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from ensemble_boxes import *
from skimage.io import imread
from torch import nn

from app.utils.duke_dbt_data import draw_box

CLASSES = ['normal', 'actionable', 'benign', 'malign', 'unknown']

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


@dataclass
class InputParameters:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ExponentialLR


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(2020)


def load_model(filename):
    print("->Loading checkpoint")
    input_parameters = torch.load(filename)
    return input_parameters.model


def load_net(filename):
    config = get_efficientdet_config('tf_efficientdet_d0')

    config.image_size = [512, 512]
    config.norm_kwargs = dict(eps=.001, momentum=.01)
    config.num_classes = 2
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint["model_state_dict"])
    return DetBenchPredict(net)


model_1 = load_model('..\\classification\\checkpoints\\model_1.pth')
model_1 = torch.jit.script(model_1)
model_1.eval()

model_2 = load_model('..\\classification\\checkpoints\\epoch_0_model_2.pth')
model_2 = torch.jit.script(model_2)
model_2.eval()

model_3 = load_net("..\\detection\\checkpoints\\model_3.pth")


def preprocess_image(selected_image):
    selected_image = np.moveaxis(selected_image, -1, 0)
    slices = np.empty(shape=[1, 3, 512, 512], dtype=np.float32)
    slices[0] = selected_image / 255.0
    slices = torch.tensor(slices)
    slices[0] = fn.normalize(slices[0], mean=mean, std=std)
    return slices


def get_label(selected_image):
    for root, dirs, files in os.walk(".\\images"):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                try:
                    image = imread(fname=image_path)
                    if (image == selected_image).all():
                        label = int(file[file.index("-") + 1])
                        return label
                except:
                    print()
    return -1


def run_wbf(predictions, image_index, image_size=512, iou_thr=0, skip_box_thr=0, weights=None):
    boxes = [(prediction[image_index]['boxes'] / (image_size - 1)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [prediction[image_index]['labels'].tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels


def make_predictions(images, score_threshold=0):
    predictions = []
    with torch.no_grad():
        dict = {'img_scale': torch.tensor(1).unsqueeze(dim=0), 'img_size': torch.tensor((512, 512)).unsqueeze(dim=0)}
        det = model_3(images, dict)
        scores = det[0].detach().cpu().numpy()[:, 4]
        boxes = det[0].detach().cpu().numpy()[:, :4]
        labels = det[0].detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > score_threshold)[0]
        boxes = boxes[indexes]
        predictions.append({
            'boxes': boxes[indexes],
            'scores': scores[indexes],
            'labels': labels[indexes]
        })
    return [predictions]


def image_classifier(selected_image):
    label = get_label(selected_image[:, :, 0])
    new_image = selected_image[:, :, 0]
    slices = preprocess_image(selected_image)
    with torch.no_grad():
        output_1 = model_1(slices)
    probabilities_1 = nn.Sigmoid()(output_1[0])[0]
    class_predicted_1 = 1
    if class_predicted_1 == 1:
        with torch.no_grad():
            output_2 = model_2(slices)
            probabilities_2 = nn.Softmax()(torch.cat((output_1[0][0].unsqueeze(dim=0)[:, 0], output_2[0][0]), dim=-1))
            # class_predicted_2 = np.argmax(probabilities_2)
            class_predicted_2 = 1
            if class_predicted_2 == 1:
                with torch.no_grad():
                    predictions = make_predictions(slices)
                    i = 0
                    boxes, scores, labels = run_wbf(predictions, image_index=i)

                    x = boxes[0][0]
                    y = boxes[0][1]
                    width = boxes[0][2] - x
                    height = boxes[0][3] - y

                    label_aux = int(labels[0]) + 1
                    selected_image = torch.tensor(selected_image).permute(2, 0, 1)
                    new_image = draw_box(image=np.array(selected_image[0]), x=int(x), y=int(y),
                                         width=int(width), height=int(height), lw=2)
    if class_predicted_1 == 0:
        probabilities_1 = nn.Softmax()(torch.cat((output_1[0][0].unsqueeze(dim=0)[:, 0:][0],
                                                  output_1[0][0].unsqueeze(dim=0)[:, 1]), dim=-1))
        confidences = {'normal': float(probabilities_1[0]),
                       'actionable': float(probabilities_1[1]),
                       CLASSES[label_aux]: float(probabilities_1[1])}
    else:
        confidences = {'normal': float(probabilities_2[0]),
                       'actionable': float(probabilities_2[1]),
                       CLASSES[label_aux]: float(probabilities_2[2])}
    return confidences, CLASSES[label], new_image


example_images = []
for root, dirs, files in os.walk(".\\images"):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)
            example_images.append(image_path)


def clear():
    return None, None, None


with gr.Blocks(theme='xiaobaiyuan/theme_brief') as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                img = gr.Image(width=512, height=512, show_download_button=True)
            with gr.Row():
                btn_clear = gr.Button("Clear")
                btn_submit = gr.Button("Submit")
        with gr.Column():
            with gr.Row():
                examples = gr.Examples(examples=example_images, inputs=img, examples_per_page=14)
            with gr.Row():
                label_predicted = gr.Label(num_top_classes=4, label="Predicted")
            with gr.Row():
                label_true = gr.Textbox(label="True label")
        btn_submit.click(image_classifier, inputs=img, outputs=[label_predicted, label_true, img])
        btn_clear.click(clear, outputs=[img, label_predicted, label_true])

demo.launch()
