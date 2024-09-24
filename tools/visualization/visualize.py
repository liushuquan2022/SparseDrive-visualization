import os, sys
#当前目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#当前目录的上一级
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
#当前目录的上上一级
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import glob
import argparse
from tqdm import tqdm
from os import path as osp
import cv2
import datetime
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender

plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = True, # True: draw history tracked boxes
    motion = True,
    map = True,
    planning = True,
)

INTERVAL = 1
visualizer_show = None
video_type = ['cam_pred', 'bev_pred', 'bev_gt', 'combine_pred', 'combine_gt']



class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
        Sources=None,
    ):
        if Sources is None:
            self.results = mmcv.load(args.result_path)
            self.out_dir = args.out_dir
        else:
            self.results = Sources
            if args.show_dir is None:
                self.out_dir = './work_dirs/vis'
            else:
                self.out_dir = args.show_dir

        self.combine_pred = os.path.join(self.out_dir, 'combine_pred')
        self.combine_gt = os.path.join(self.out_dir, 'combine_gt')
        os.makedirs(self.combine_pred, exist_ok=True)
        os.makedirs(self.combine_gt, exist_ok=True)

        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)

        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]['img_bbox']

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)
    
    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)

        merge_image = cv2.hconcat([cam_image, bev_image])
        imgs_path = glob.glob(os.path.join(self.combine_pred, '*.jpg'))
        index = len(imgs_path)
        save_path = os.path.join(self.combine_pred, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

        merge_image = cv2.hconcat([cam_image, bev_gt])
        imgs_path = glob.glob(os.path.join(self.combine_gt, '*.jpg'))
        index = len(imgs_path)
        save_path = os.path.join(self.combine_gt, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, s_type='combine_gt',fps=12, downsample=4):
        video_dir = os.path.join(self.out_dir, s_type)
        imgs_path = glob.glob(os.path.join(video_dir, '*.jpg'))
        video_filename = s_type + '.mp4'

        imgs_path = sorted(imgs_path)
        out = None
        out_path = os.path.join(self.out_dir, video_filename)
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)

            if out is None:
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            out.write(img)
            # video_type = ['cam_pred', 'bev_pred', 'bev_gt', 'combine_pred', 'combine_gt']
            if s_type == 'combine_pred':
                cv2.namedWindow(s_type, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(s_type, width, height)
                cv2.imshow(s_type, img)
                if cv2.waitKey(1) == ord('q'):
                   exit()

        if out is not None:
            out.release()

    def show_video(self, s_type='combine_gt',fps=12, downsample=4):
        video_dir = os.path.join(self.out_dir, s_type)
        imgs_path = glob.glob(os.path.join(video_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)

        img_path = imgs_path[len(imgs_path)-1]
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height //
                         downsample), interpolation=cv2.INTER_AREA)
        height, width, channel = img.shape

        # video_type = ['cam_pred', 'bev_pred', 'bev_gt', 'combine_pred', 'combine_gt']
        if s_type == 'combine_gt':
            cv2.namedWindow(s_type, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(s_type, width, height)
            cv2.imshow(s_type, img)
            if cv2.waitKey(1) == ord('q'):
               exit()


def visualizer_show_window(result, vis_args):
    # Display to result
    visualizer_show = Visualizer(vis_args, plot_choices, result)
    visualizer_show.add_vis(0)
    for v_type in video_type:
        visualizer_show.show_video(v_type)

def visualizer_save_video(out_dir, fps=12, downsample=4):
    for idx in tqdm(range(0, len(video_type), INTERVAL)):
        v_type = video_type[idx]
        video_dir = os.path.join(out_dir, v_type)
        imgs_path = glob.glob(os.path.join(video_dir, '*.jpg'))
        video_filename = v_type + '.mp4'
        if len(imgs_path)==0:
            continue

        imgs_path = sorted(imgs_path)
        out = None
        out_path = os.path.join(out_dir, video_filename)
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)

            if out is None:
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            out.write(img)

        if out is not None:
            out.release()

def visualizer_args():
    vis_parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    vis_parser.add_argument('config', help='config file path')
    vis_parser.add_argument('--result-path',
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    vis_parser.add_argument(
        '--out-dir', 
        default='./show_dirs/vis',
        help='directory where visualize results will be saved')
    vis_args = vis_parser.parse_args()

    return vis_args

def main():
    vis_args = visualizer_args()

    # set out dir
    work_timestr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    config_eval = osp.splitext(osp.basename(vis_args.config))[0]
    config_eval = config_eval + '_' + work_timestr
    out_dir = osp.join('./show_dirs', config_eval)
    mmcv.mkdir_or_exist(osp.abspath(out_dir))
    vis_args.out_dir = out_dir
    print('show_dirs: ',vis_args.out_dir)

    visualizer_main = Visualizer(vis_args, plot_choices, None)
    START = 0
    #END = 380
    END = len(visualizer_main.results)
    for idx in tqdm(range(START, END, INTERVAL)):
        if idx > END:
            break
        visualizer_main.add_vis(idx)

    # Video File
    for v_type in video_type:
        visualizer_main.image2video(v_type)

if __name__ == '__main__':
    main()