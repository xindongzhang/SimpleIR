import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from models.ecbsr import ECBSR
import os
import numpy as np
import cv2
import math
import glob
import PIL.Image as pil_image
import torch.backends.cudnn as cudnn
import yaml

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 255. + 408.583 * img[..., 2] / 255. - 222.921
        g = 298.082 * img[..., 0] / 255. - 100.291 * img[..., 1] / 255. - 208.120 * img[..., 2] / 255. + 135.576
        b = 298.082 * img[..., 0] / 255. + 516.412 * img[..., 1] / 255. - 276.836
    else:
        r = 298.082 * img[0] / 255. + 408.583 * img[2] / 255. - 222.921
        g = 298.082 * img[0] / 255. - 100.291 * img[1] / 255. - 208.120 * img[2] / 255. + 135.576
        b = 298.082 * img[0] / 255. + 516.412 * img[1] / 255. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 255.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 255.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 255.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 255.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 255.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 255.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr

def calc_psnr(img1, img2):
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config file of the model')
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    pretrain_model_path = './experiments/imdn-x4-2022-0405-0941/models/model_x{}_{}.pt'.format(args.scale, 210)
    # pretrain_model_path = './experiments/edsr-x4-r16c64-2022-0405-0941/models/model_x{}_{}.pt'.format(args.scale, 210)
    if args.model == 'ecbsr':
        from models.ecbsr import ECBSR as Network
        model = Network(
            module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, 
            with_idt=args.idt_ecbsr, act_type=args.act_type, 
            scale=args.scale, colors=args.colors, step_wise=args.step_wise
        ).to(device)
        print("select ecbsr as the base architecture! ")
    elif args.model == 'imdn':
        from models.imdn import IMDN as Network
        model = Network(in_nc=args.colors, out_nc=args.colors).to(device)
        print("select imdn as the base architecture! ")
    elif args.model == 'edsr':
        from models.edsr import EDSR as Network
        model = Network(args=args).to(device)
        print("select edsr as the base architecture! ")
    else:
        raise ValueError('not supported model type!')

    model.load_state_dict(torch.load(pretrain_model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    image_files = glob.glob("/home/xindongzhang/SR_datasets/benchmark/Urban100/LR_bicubic/X4/*.png")
    # image_files = glob.glob("/home/xindongzhang/SR_datasets/benchmark/Set14/LR_bicubic/X4/*.png")
    # image_files = glob.glob("/home/xindongzhang/SR_datasets/benchmark/B100/LR_bicubic/X4/*.png")

    avg_psnr = 0.0
    for idx, image_file in enumerate(image_files):
        # output_dir = './results/{}_x{}_u100/'.format(args.model, args.scale)
        output_dir = './results/{}_x{}_u100/'.format(args.model, args.scale)
        output_dir_bicubic = './results/bicubic_u100/'.format(args.model, args.scale)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir_bicubic):
            os.makedirs(output_dir_bicubic)

        image = pil_image.open(image_file).convert('RGB')
        image_width = image.width * args.scale
        image_height = image.height * args.scale

        hr_image_file = image_file.replace('x4', '')
        hr_image_file = hr_image_file.replace('LR_bicubic/X4', 'HR')
        hr_img = pil_image.open(hr_image_file).convert('RGB')
        hr_width = (hr_img.width // args.scale) * args.scale
        hr_height = (hr_img.height // args.scale) * args.scale
        hr_img = hr_img.crop((0,0,hr_width,hr_height))

        tag = image_file.split('/')[-1]

        bicubic = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

        lr, _ = preprocess(image, device)
        hr, _ = preprocess(hr_img, device)
        br, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 255.0)

        psnr = calc_psnr(hr[0, 0, 4:-4, 4:-4], preds[0, 0, 4:-4, 4:-4])
        avg_psnr += psnr
        # psnr_br = calc_psnr(hr, br)

        preds = preds.cpu().numpy().squeeze(0).squeeze(0)


        sr = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        sr = np.clip(convert_ycbcr_to_rgb(sr), 0.0, 255.0).astype(np.uint8)
        sr = pil_image.fromarray(sr)
        # log = '{}, {}, {}, {}'.format(idx, tag, psnr_br.cpu().numpy(), psnr.cpu().numpy())
        # file_log.write(log + '\n')
        # print(log)

        print(idx, tag, psnr)
        sr.save(os.path.join(output_dir, tag))
        bicubic.save(os.path.join(output_dir_bicubic, tag))
    # file_log.close()
    print(avg_psnr/len(image_files))