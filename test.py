# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
from sklearn.metrics import auc

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset

# Model
from model.HFMNet import HFMNet


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Load model 
        if args.model   == 'HFMNet':
            model = HFMNet()


        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load('result/' + args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        Epoch = checkpoint['epoch']

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision, f1_score = self.ROC.get()
                Accuracy = auc(false_positive_rate, ture_positive_rate)
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids)) 
            scio.savemat(dataset_dir + '/' +  'value_result'+ '/' + args.st_model + '_evulate.mat', {'mIOU': mean_IOU,'Recall': recall,'Precision': precision,'Accuracy':Accuracy,'TPR': ture_positive_rate,'FPR': false_positive_rate,'PD': PD,'FA': FA,'F1-Score': f1_score})
            save_result_for_test(dataset_dir, args.st_model, Epoch, mean_IOU, recall, precision, FA, PD, ture_positive_rate, false_positive_rate, f1_score)

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
