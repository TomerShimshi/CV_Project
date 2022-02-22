"""Define your architecture here."""
import torch
from trainer import LoggingParameters, Trainer
#from models import 
from utils import load_dataset, get_nof_params
import torchvision.models as models
import torchvision
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    #model = SimpleNet()
    #vgg = models.vgg16(pretrained=True)
    #inception = models.inception_v3(pretrained=True)
    resnet= models.resnet18(pretrained=True)
    #print(resnet)

    
    model= nn.Sequential( nn.Upsample((224,224), mode='nearest'),resnet,nn.BatchNorm1d(1000),
    nn.Linear(1000,512),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    #nn.InstanceNorm2d(3),
    nn.Linear(512,256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(256,64),
    nn.BatchNorm1d(64),
    nn.LeakyReLU(),
    nn.Linear(64,2))
    # load your model using exactly this line (don't change it):


    
    ################## Train The Model_need to delete later
    train_dataset = load_dataset('fakes_dataset', 'train')
    trainer = Trainer(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                      criterion=nn.CrossEntropyLoss(), batch_size=32,
                      
                      train_dataset=train_dataset,
                      validation_dataset=train_dataset,
                      test_dataset=train_dataset)
    for i in range(3):
        trainer.train_one_epoch()
    checkpoint_filename= 'checkpoints/bonus_model.pt'
    # Save checkpoint
    
    print(f'Saving checkpoint {checkpoint_filename}')
    state = {
        'model': model.state_dict(),
        
    }
    torch.save(state, checkpoint_filename)
    
   
    
    #model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
