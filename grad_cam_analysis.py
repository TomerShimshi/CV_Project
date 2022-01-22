"""Show network train graphs and analyze training results."""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model

### I ADDED THIS###############
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoint path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='checkpoints/synthetic_dataset_XceptionBased_Adam.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='synthetic_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def get_grad_cam_visualization(test_dataset: torch.utils.data.Dataset,
                               model: torch.nn.Module): #-> tuple[np.ndarray,
                                                       #         torch.tensor]:
    """Return a tuple with the GradCAM visualization and true class label.

    Args:
        test_dataset: test dataset to choose a sample from.
        model: the model we want to understand.

    Returns:
        (visualization, true_label): a tuple containing the visualization of
        the conv3's response on one of the sample (256x256x3 np.ndarray) and
        the true label of that sample (since it is an output of a DataLoader
        of batch size 1, it's a tensor of shape (1,)).
    """
    """INSERT YOUR CODE HERE, overrun return."""

    if (torch.cuda.is_available()):
        device='cpu'
    else:
        device= 'gpu'
    #dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    target_layers = [model.conv3]
    input_tensor = DataLoader(test_dataset, batch_size=1, shuffle=True)# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model, target_layers=target_layers, use_cuda=False)#torch.cuda.is_available())
    
    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...
    
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    inputs, targets = next(iter(input_tensor))
    # for every image in the batch.
    #temp = model(inputs)
    print('shuff')
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=inputs)
    
    # In this example grayscale_cam has only one image in the batch:
    inputs=(inputs[0])
    inputs = np.transpose(inputs, (1, 2, 0))
    grayscale_cam = grayscale_cam[0, :]
    inputs=inputs.cpu().numpy()#(inputs[0].cpu().numpy().astype(np.float32))#/255.0)#.astype(np.float32)
    
    #inputs=np.rot90(inputs)#np.reshape(inputs,(256,256,3))
    max_val= np.amax(inputs)
    min_val= np.amin(inputs)
    inputs = ((inputs-min_val)/(float(max_val- min_val)))#np.clip(inputs, 0, 1)
    #normelaize the image
    #plt.imshow(inputs)
    #plt.show()
    #print('inputs type{}'.format(type(inputs)))
    visualization = show_cam_on_image(inputs, grayscale_cam, use_rgb=True)

    return visualization,targets #np.random.rand(256, 256, 3), torch.randint(0, 2, (1,))


def main():
    """Create two GradCAM images, one of a real image and one for a fake
    image for the model and dataset it receives as script arguments."""
    args = parse_args()
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])

    model.eval()
    seen_labels = []
    while len(set(seen_labels)) != 2:
        visualization, true_label = get_grad_cam_visualization(test_dataset,
                                                               model)
        grad_cam_figure = plt.figure()
        plt.imshow(visualization)
        title = 'Fake Image' if true_label == 1 else 'Real Image'
        plt.title(title)
        seen_labels.append(true_label.item())
        grad_cam_figure.savefig(
            os.path.join(FIGURES_DIR,
                         f'{args.dataset}_{args.model}_'
                         f'{title.replace(" ", "_")}_grad_cam.png'))


if __name__ == "__main__":
    main()
