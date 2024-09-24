import os, sys
#当前目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#当前目录的上一级
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

import datetime
import argparse
import mmcv
from os import path as osp

import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmdet.apis import single_gpu_test, multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
from mmdet.core import encode_mask_results

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test

from visualization.visualize import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMDet test (and eval) a model"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--is_save", action="store_true", help="save to result file in pickle format")
    parser.add_argument(
        "--fuse_conv",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format_only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--visual", action="store_true", help="visual results")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show_dir", help="directory where results will be saved")
    parser.add_argument(
        "--show_type",
        nargs="+",
        action=DictAction,
        default=['combine_gt'],
        help="['cam_pred', 'bev_pred', 'bev_gt', 'combine_pred', 'combine_gt']",
    )

    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--analysis", action="store_true")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def visual_results(model, data_loader, work_dir, args):
    results = []
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)
        # 可视化
        if args.show_dir is None:
            args.show_dir = osp.join(work_dir, 'vis')

        if i == 0:
            visualizer_show = Visualizer(args, plot_choices, result)
        else:
            visualizer_show.results = result

        visualizer_show.add_vis(0)
        for v_type in args.show_type:
            if v_type in video_type:
                visualizer_show.show_video(v_type)

        for _ in range(batch_size):
            prog_bar.update()

    # 保存可视化视频
    visualizer_save_video(os.path.join(work_dir, 'vis'))
    return results

def main():
    args = parse_args()

    assert (
        args.eval or args.format_only or args.show or args.show_dir
    ), (
        "Please specify at least one operation (eval/format/show)"
        ' with the argument "--eval", "--format_only", "--show" or "--show_dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline
            )
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # set work dir
    work_timestr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        config_eval = osp.splitext(osp.basename(args.config))[0]
        if args.eval:
            config_eval = config_eval + '_' + args.eval[0]
        config_eval = config_eval + '_' + work_timestr
        cfg.work_dir = osp.join('./work_dirs', config_eval)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir
    print('work_dirs: ',cfg.work_dir)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    if distributed:
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
    else:
        data_loader = build_dataloader_origin(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if torch.cuda.is_available():
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cuda')
    else:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv:
        model = fuse_conv_bn(model)

    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if args.result_file is not None:
        # outputs = torch.load(args.result_file)
        outputs = mmcv.load(args.result_file)
    elif not distributed:
        model = MMDataParallel(model, device_ids=[0])
        if args.visual:
            outputs = visual_results(model, data_loader, cfg.work_dir, args)
        else:
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

        out_filename = osp.join(cfg.work_dir, 'results.pkl')
        print(f"\nwriting results to {out_filename}")
        mmcv.dump(outputs, out_filename)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            )
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.is_save and not args.visual:
            out_filename = osp.join(cfg.work_dir, 'results.pkl')
            print(f"\nwriting results to {out_filename}")
            mmcv.dump(outputs, out_filename)

        # analysis
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        elif args.eval and args.analysis:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(eval_kwargs)
            results_dict = dataset.evaluate(outputs, **eval_kwargs)
            print(results_dict)

if __name__ == "__main__":
    # use fork workers_per_gpu can be > 1
    # torch.multiprocessing.set_start_method("fork")

    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
