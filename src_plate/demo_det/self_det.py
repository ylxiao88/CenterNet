from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
#from progress.bar import Bar
import time
import torch

#from models.model import create_model, load_model
#from utils.image import get_affine_transform
#from utils.debugger import Debugger
from dlav0 import get_pose_net as get_dlav0
from decode import ctdet_decode
#from utils.post_process import ctdet_post_process


class CtdetDetector(object):
  def __init__(self, opt):
    #if opt.gpus[0] >= 0:
    #  opt.device = torch.device('cuda')
    #else:
    #  opt.device = torch.device('cpu')
    opt.device = torch.device('cuda')

    print('Creating model...')
    #self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    #self.model = get_dlav0(num_layers=34, heads=opt.heads, head_conv=opt.head_conv)
    self.model = get_dlav0(num_layers=34, heads={'hm':1,'wh':2,'reg':2}, head_conv=256)
    #self.model = load_model(self.model, opt.load_model)
    checkpoint = torch.load(opt.load_model)
    self.model.load_state_dict(checkpoint['state_dict'])

    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    #self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale=1, meta=None):
    height, width = image.shape[0:2]
    #new_height = int(height * scale)
    #new_width  = int(width * scale)

    #if self.opt.fix_res:
    inp_height, inp_width = self.opt.input_h, self.opt.input_w
    #c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    #s = max(height, width) * 1.0
    #else:
    #  inp_height = (new_height | self.opt.pad) + 1
    #  inp_width = (new_width | self.opt.pad) + 1
    #  c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
    #  s = np.array([inp_width, inp_height], dtype=np.float32)

    #trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    #resized_image = cv2.resize(image, (new_width, new_height))
    #inp_image = cv2.warpAffine(
    #  resized_image, trans_input, (inp_width, inp_height),
    #  flags=cv2.INTER_LINEAR)
    inp_image = cv2.resize(image,(inp_width, inp_height))
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    #if self.opt.flip_test:
    #  images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {
            #'c': c, 's': s,
            #'out_height': inp_height // self.opt.down_ratio,
            #'out_width': inp_width // self.opt.down_ratio}
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta

  #def process(self, images, return_time=False):
  #  raise NotImplementedError
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      #reg = output['reg'] if self.opt.reg_offset else None
      reg = output['reg']
      #if self.opt.flip_test:
      #  hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
      #  wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
      #  reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      #dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets




  def post_process(self, dets, meta, image, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    #dets = ctdet_post_process(
    #    dets.copy(), [meta['c']], [meta['s']],
    #    meta['out_height'], meta['out_width'], self.opt.num_classes)
    dets = self.ctdet_post_process(dets.copy(), meta, image)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      #dets[0][j][:, :4] /= scale
    return dets[0]

  def ctdet_post_process(self, dets, meta, image):
    out_h = float(meta['out_height'])
    out_w = float(meta['out_width'])
    ih,iw,_ = image.shape

    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i,:, 0] *= float(iw)/out_w
        dets[i,:, 1] *= float(ih)/out_h
        dets[i,:, 2] *= float(iw)/out_w
        dets[i,:, 3] *= float(ih)/out_h
        classes = dets[i, :, -1]
        inds = (classes == 0)
        top_preds[1] = np.concatenate([
                            dets[i, inds, :4].astype(np.float32),
                            dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      #if len(self.scales) > 1 or self.opt.nms:
      #   soft_nms(results[j], Nt=0.5, method=2)
    #scores = np.hstack(
    #  [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    #if len(scores) > self.max_per_image:
    #  kth = len(scores) - self.max_per_image
    #  thresh = np.partition(scores, kth)[kth]
    #  for j in range(1, self.num_classes + 1):
    #    keep_inds = (results[j][:, 4] >= thresh)
    #    results[j] = results[j][keep_inds]
    return results

  #def show_results(self, debugger, image, results):
  def show_results(self, image, results):
    #debugger.add_img(image, img_id='ctdet')
    #for j in range(1, self.num_classes + 1):
    for bbox in results[1]:
        if bbox[4] > self.opt.vis_thresh:
            #debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
            conf = bbox[4]
            bbox = np.array(bbox[:4], dtype=np.int32)
            c = (0, 0, 255)
            cv2.rectangle(
                image, (bbox[0],bbox[1]), (bbox[2], bbox[3]), c, 2
            )
            # text
            txt = '%.2f'%(conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.8, 2)[0]
            cv2.rectangle(
                image, (bbox[0], bbox[1]-cat_size[1]-2), (bbox[0]+cat_size[0], bbox[1]-2),c,-1)
            cv2.putText(
                image, txt, (bbox[0], bbox[1]-2), font, 0.8, (0,0,0), thickness=1, lineType=cv2.LINE_AA
            )

    #debugger.show_all_imgs(pause=self.pause)
    cv2.imshow('res', image)
    cv2.waitKey(1000)



  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError


  def run(self, image_or_path_or_tensor, meta=None):
    #load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    #merge_time, tot_time = 0, 0
    #debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
    #debugger = Debugger(dataset=self.opt.dataset,
    #                    theme=self.opt.debugger_theme)
    #start_time = time.time()
    #pre_processed = False
    #if isinstance(image_or_path_or_tensor, np.ndarray):
    #  image = image_or_path_or_tensor
    #elif type(image_or_path_or_tensor) == type (''):
    #  image = cv2.imread(image_or_path_or_tensor)
    image = cv2.imread(image_or_path_or_tensor)
    #else:
    #  image = image_or_path_or_tensor['image'][0].numpy()
    #  pre_processed_images = image_or_path_or_tensor
    #  pre_processed = True

    #loaded_time = time.time()
    #load_time += (loaded_time - start_time)

    detections = []
    #for scale in self.scales:
    #scale_start_time = time.time()
    #if not pre_processed:
    #  images, meta = self.pre_process(image, scale, meta)
    images, meta = self.pre_process(image, 1, meta)
    #else:
    #  # import pdb; pdb.set_trace()
    #  images = pre_processed_images['images'][scale][0]
    #  meta = pre_processed_images['meta'][scale]
    #  meta = {k: v.numpy()[0] for k, v in meta.items()}
    images = images.to(self.opt.device)
    torch.cuda.synchronize()
    #pre_process_time = time.time()
    #pre_time += pre_process_time - scale_start_time

    #output, dets, forward_time = self.process(images, return_time=True)
    output, dets = self.process(images)

    torch.cuda.synchronize()
    #net_time += forward_time - pre_process_time
    #decode_time = time.time()
    #dec_time += decode_time - forward_time

    #if self.opt.debug >= 2:
    #  self.debug(debugger, images, dets, output, scale)

    #dets = self.post_process(dets, meta, scale)
    dets = self.post_process(dets, meta, image)
    torch.cuda.synchronize()
    #post_process_time = time.time()
    #post_time += post_process_time - decode_time

    detections.append(dets)

    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    #end_time = time.time()
    #merge_time += end_time - post_process_time
    #tot_time += end_time - start_time

    #if self.opt.debug >= 1:
    #self.show_results(debugger, image, results)
    self.show_results(image, results)

    #return {'results': results, 'tot': tot_time, 'load': load_time,
    #        'pre': pre_time, 'net': net_time, 'dec': dec_time,
    #        'post': post_time, 'merge': merge_time}
    return results
