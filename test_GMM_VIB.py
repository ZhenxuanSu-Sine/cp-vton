# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from networks_baseline import GMM_baseline

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def test_gmm_VIB(opt, test_loader_im, test_loader_c, baseline_model, models, board):
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    baseline_model.cuda()
    baseline_model.eval()
    for model in models:
        model.cuda()
        model.eval()
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    criterionL1 = nn.L1Loss()

    for (step_im, inputs_im), (step_c, inputs_c) in zip(enumerate(test_loader_im.data_loader),
                                                        enumerate(test_loader_c.data_loader)):
        iter_start_time = time.time()

        c_names = inputs_c['c_name']
        im = inputs_im['image'].cuda()
        im_pose = inputs_im['pose_image'].cuda()
        im_h = inputs_im['head'].cuda()
        shape = inputs_im['shape'].cuda()
        agnostic = inputs_im['agnostic'].cuda()
        c = inputs_im['cloth'].cuda()
        cm = inputs_im['cloth_mask'].cuda()
        im_c = inputs_im['parse_cloth'].cuda()
        im_g = inputs_im['grid_image'].cuda()

        # warped_clothes = []
        titles = ['image', 'cloth', 'baseline']
        images = [im[0, :, :, :], c[0, :, :, :]]
        agnostic = agnostic[0: 1, :, :, :]
        im = im[0: 1, :, :, :]
        c = c[0: 1, :, :, :]
        grid, theta = baseline_model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        images.append(warped_cloth[0, :, :, :])
        for model in models:
            grid, theta, VIB_loss = model(im, c)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            titles.append(model.name)
            images.append(warped_cloth[0, :, :, :])
            print("VIB loss of %s: %4f" % (model.name, VIB_loss))
        # visuals = [ [im_h, shape, im_pose],
        #            [c, warped_cloth, im_c],
        #            [warped_grid, (warped_cloth+im)*0.5, im]]
        for i in range(len(images)):
            img = images[i].transpose(2, 0).transpose(1, 0).cpu()
            # print(img[0, 0, 0])
            img = transforms.Normalize((-1,), (2,))(img)
            # print(img[0, 0, 0])
            plt.subplot((len(images) + 2) // 2, 2, i + 1), plt.imshow(img)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
        input('next pair')
        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        #
        # if (step+1) % opt.display_count == 0:
        #     board_add_images(board, 'combine', visuals, step+1)
        #     L1_loss = criterionL1(warped_cloth, im_c)
        #     t = time.time() - iter_start_time
        #     print('step: %8d, time: %.3f, L1_loss: %4f, VIB_loss: %4f' % (step+1, t, L1_loss.item(), VIB_loss), flush=True)
        #


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2 * cm - 1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step + 1, t), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader_im = CPDataLoader(opt, train_dataset)
    train_loader_c = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train
    if opt.stage == 'GMM':
        checkpoints = ['gmm_with_VIB_0', 'gmm_with_VIB_1e-2', 'gmm_with_VIB_1e-3', 'gmm_with_VIB_1e-5']
        models = []
        for checkpoint in checkpoints:
            model = GMM(opt)
            load_checkpoint(model, "checkpoints/%s/gmm_final.pth" % checkpoint)
            models.append(model)
        baseline = GMM_baseline(opt)
        load_checkpoint(baseline, "checkpoints/gmm_train_new/gmm_final.pth")
        with torch.no_grad():
            test_gmm_VIB(opt, train_loader_im, train_loader_c, baseline, models, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
