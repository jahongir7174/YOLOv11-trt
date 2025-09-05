import os
import random
import time
import warnings
from argparse import ArgumentParser

import cv2
import numpy
import torch
import yaml

from utils import util

warnings.filterwarnings("ignore")


def latency(device, engine_path):
    engine = util.TRTModule(engine_path, device)

    # warming up
    import time
    x = torch.randn((1, 3, 640, 640), dtype=torch.float16, device=device)
    for _ in range(10):
        engine(x)
    n = 10_000
    start = time.perf_counter()
    for _ in range(n):
        engine(x)
    end = time.perf_counter()

    fps = int(1 / ((end - start) / n))
    print('FPS:', fps)


def run(args, params, device, engine_path):
    engine = util.TRTModule(engine_path, device)

    engine.set_desired(['num', 'boxes', 'scores', 'labels'])

    names = list(params['names'].values())
    colors = {cls: [random.randint(0, 255) for _ in range(3)]
              for i, cls in enumerate(names)}

    root = './data'
    filenames = os.listdir(f'{root}')
    filenames.sort()

    for filename in filenames:
        # Capture frame-by-frame
        frame = cv2.imread(f"{root}/{filename}")
        image = frame.copy()
        shape = image.shape[:2]

        r = args.input_size / max(shape[0], shape[1])
        if r != 1:
            resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
        height, width = image.shape[:2]

        # Scale ratio (new / old)
        r = min(1.0, args.input_size / height, args.input_size / width)

        # Compute padding
        pad = int(round(width * r)), int(round(height * r))
        w = (args.input_size - pad[0]) / 2
        h = (args.input_size - pad[1]) / 2

        if (width, height) != pad:  # resize
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

        # Convert HWC to CHW, BGR to RGB
        x = image.transpose((2, 0, 1))[::-1]
        x = numpy.ascontiguousarray(x)
        x = torch.from_numpy(x)
        x = x.unsqueeze(dim=0)
        x = x.float()
        x = x.cuda()
        x = x / 255
        # Inference
        y = engine(x)
        num, boxes, scores, labels = y[0][0], y[1][0], y[2][0], y[3][0]
        num = num.item()
        if num != 0:
            # check score negative
            scores[scores < 0] = 1 + scores[scores < 0]
            boxes = boxes[:num]
            scores = scores[:num]
            labels = labels[:num]
            if boxes.numel() != 0:
                boxes[:, [0, 2]] -= w  # x padding
                boxes[:, [1, 3]] -= h  # y padding
                boxes[:, :4] /= min(height / shape[0], width / shape[1])

                boxes[:, 0].clamp_(0, shape[1])  # x1
                boxes[:, 1].clamp_(0, shape[0])  # y1
                boxes[:, 2].clamp_(0, shape[1])  # x2
                boxes[:, 3].clamp_(0, shape[0])  # y2

                for box, score, label in zip(boxes, scores, labels):
                    box = box.round().int().tolist()
                    cls = names[int(label)]

                    text = f'{cls}:{score:.3f}'
                    x1, y1, x2, y2 = box

                    (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.8, 1)
                    _y1 = min(y1 + 1, frame.shape[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
                    cv2.rectangle(frame, (x1, _y1), (x1 + _w, _y1 + _h + _bl),
                                  (0, 0, 255), -1)
                    cv2.putText(frame, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), 2)
        os.makedirs(name='./outputs', exist_ok=True)
        cv2.imwrite(f'./outputs/{filename}', frame)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--engine', action='store_true')
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--run', action='store_true')

    args = parser.parse_args()

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    device = torch.device('cuda:0')
    onnx_path = './weights/v11_n.onnx'
    model_path = './weights/v11_n.pt'
    engine_path = './weights/v11_n.engine'

    if args.onnx:
        util.to_onnx(device, model_path, onnx_path)
    time.sleep(1)
    if args.engine:
        util.to_engine(device, onnx_path, engine_path)
    time.sleep(1)
    if args.benchmark:
        latency(device, engine_path)
    time.sleep(1)
    if args.run:
        run(args, params, device, engine_path)

    # Clean
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
