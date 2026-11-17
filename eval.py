import os
import torch
import argparse
import logging
from lib import evaluation


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='coco', help='coco or f30k')
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/01_Code/00_Laps/data', help='the path of dataset')
    parser.add_argument('--f30k_img_path', type=str, default='/home/ubuntu/00_Data/F30k/images', help='the path of f30k images')
    parser.add_argument('--coco_img_path', type=str, default='/home/ubuntu/00_Data/COCO/images/', help='the path of coco images')
    parser.add_argument('--save_results', type=int, default=0, help='whether save the results')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.dataset == 'coco':
        weights_bases = [    
            'runs',
        ]
    else:
        weights_bases = [
            'runs',
        ]

    for base in weights_bases:

        # logging.basicConfig(filename=os.path.join(base, 'eval_extra.log'), filemode='w', 
        #                     format='%(asctime)s %(message)s', level=logging.DEBUG, force=True)
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.info('Evaluating {}...'.format(base))
        model_name = 'model_best_coco.pth' if opt.dataset == 'coco' else 'model_best.pth'
        model_path = os.path.join(base, model_name)
        
        # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset)) if opt.save_results else None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True,
                                coco_img_path=opt.coco_img_path)
            # Evaluate COCO 5K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path,
                                coco_img_path=opt.coco_img_path)
        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path,
                                f30k_img_path=opt.f30k_img_path)


if __name__ == '__main__':
    
    main()
